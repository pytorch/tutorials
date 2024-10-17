import torch
import logging

######################################################################
#
# Setup enhanced logging for devices that don't support torch.compile
#
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check device capabilities and handle cases where torch.compile is not supported
def check_device_and_log():
    """Check device capability and log whether torch.compile is supported."""
    if torch.cuda.is_available():
        # Get CUDA device capability
        capability = torch.cuda.get_device_capability()
        logger.info(f"CUDA Device Capability: {capability}")

        if capability < (7, 0):
            # Log the reason why torch.compile is not supported
            logger.warning(
                "torch.compile is not supported on devices with a CUDA capability less than 7.0."
            )
            return False
        logger.info("Device supports torch.compile.")
        return True
    else:
        logger.info("No CUDA device found. Using CPU.")
        return False


# Function to apply torch.compile only if supported
def fn(x, y):
    """Simple function to add two tensors and return the result."""
    z = x + y
    return z + 2


def run_example_with_logging():
    """Run example using torch.compile if supported, with logging."""
    # Check if the device supports torch.compile
    compile_supported = check_device_and_log()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = (
        torch.ones(2, 2, device=device),
        torch.zeros(2, 2, device=device),
    )

    if compile_supported:
        logger.info("Compiling the function with torch.compile...")
        compiled_fn = torch.compile(fn)
    else:
        logger.info("Running the function without compilation...")
        compiled_fn = fn  # Use the uncompiled function

    # Print separator and reset dynamo between each example
    def separator(name):
        print(f"==================={name}=========================")
        torch._dynamo.reset()

    separator("Dynamo Tracing")
    torch._logging.set_logs(dynamo=logging.DEBUG)
    compiled_fn(*inputs)

    separator("Traced Graph")
    torch._logging.set_logs(graph=True)
    compiled_fn(*inputs)

    separator("Fusion Decisions")
    torch._logging.set_logs(fusion=True)
    compiled_fn(*inputs)

    separator("Output Code")
    torch._logging.set_logs(output_code=True)
    compiled_fn(*inputs)

    separator("")


if __name__ == "__main__":
    run_example_with_logging()
