from dataclasses import dataclass
import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, SplitPoint, PipelineStage, ScheduleGPipe

# torchrun --standalone --nnodes 1 --nproc_per_node 2 pipelining_tut.py

rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
dist.init_process_group()

pp_group = dist.new_group()
stage_index = rank
num_stages = world_size

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = 10000

class Transformer(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        # Using a ModuleDict lets us delete layers witout affecting names,
        # ensuring checkpoints will correctly save and load.
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = nn.TransformerDecoderLayer(model_args.dim, model_args.n_heads)

        self.norm = nn.LayerNorm(model_args.dim)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size)

    def forward(self, tokens: torch.Tensor):
        # Handling layers being 'None' at runtime enables easy pipeline splitting
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        for layer in self.layers.values():
            h = layer(h, h)

        h = self.norm(h) if self.norm else h
        output = self.output(h).float() if self.output else h
        return output



# Manual usage
def manual_model_split(model) -> PipelineStage:
    # To be implemented
    stage = None
    return stage

# Tracer usage (pipeline API)
def tracer_model_split(model) -> PipelineStage:   
    x = torch.ones(32, 500, dtype=torch.long)
    pipe = pipeline(
        module=model,
        mb_args=(x,),
        split_spec={
            "layers.4": SplitPoint.BEGINNING,
        }
    )
    stage = pipe.build_stage(stage_index, device, pp_group)
    return stage

if __name__ == "__main__":
    model = Transformer(ModelArgs())
    if rank == 0:
        print(model)
        stage = tracer_model_split(model)
    
        print(stage)

        schedule = ScheduleGPipe(stage, n_microbatches=4)
        x = torch.ones(32, 500, dtype=torch.long)

        output = schedule.step(x)
        print(output)
