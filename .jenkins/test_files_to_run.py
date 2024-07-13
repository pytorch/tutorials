#!/usr/bin/env python
from get_files_to_run import get_all_files, calculate_shards
from unittest import TestCase, main
from functools import reduce

class TestSharding(TestCase):
    def test_no_sharding(self):
        all_files=get_all_files()
        sharded_files = calculate_shards(all_files, 1)
        self.assertSetEqual(set(all_files), set(sharded_files[0]))

    def test_sharding(self, num_shards=20):
        all_files=get_all_files()
        sharded_files = map(set, calculate_shards(all_files, num_shards))
        self.assertSetEqual(set(all_files), reduce(lambda x,y: x.union(y), sharded_files, set()))



if __name__ == "__main__":
    main()
