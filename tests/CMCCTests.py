import os, sys

import numpy as np

dev_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))
root = os.path.join(dev_root, 'cmcc_reactions')
deps = os.path.join(dev_root, 'dependencies')
sys.path.insert(0, root)
sys.path.insert(0, deps)
# os.chdir(root)

import os.path
import unittest
import itertools
import pprint
import tempfile as tf
import cmcc_reactions.reaction_data_schema as schema
import cmcc_reactions.data_analysis_tools as thc_tools

__all__ = [
    "CMCCTests"
]

class CMCCTests(unittest.TestCase):

    @unittest.skip
    def test_DataSerializer(self):
        self.skipTest('...')
        tree = {
            'a':{
                'b':{'c':np.array([[1, 2, 3], [4, 5, 6]]), 'e':[3, 4, 5]},
                'd':{'c':np.array([1, 2])},
                'c':10
            },
            'f':{'g':[1], 'c':[6, 7], 'x':np.random.rand(100, 100)}
        }
        serialized = schema.compress_tree(tree)
        # pprint.pprint(serialized)
        deserialized = schema.decompress_tree(serialized)
        # pprint.pprint(deserialized)


        with tf.TemporaryDirectory() as td:
            js_file = os.path.join(td, 'tree.json')
            schema.write_tree(js_file, tree, mode='json')

            tmp_file = os.path.join(td, 'tree.npz')
            schema.write_tree(tmp_file, tree)

            tree2 = schema.read_tree(tmp_file)
            tree3 = schema.read_tree(js_file, mode='json')
            # pprint.pprint(tree2)
            # pprint.pprint(tree3)

            print(os.path.getsize(tmp_file))
            print(os.path.getsize(js_file))

    @unittest.skip
    def test_DataValidator(self):
        data = {
            'reaction_1': {
                'system_1' : {
                    'faked': {
                        'atoms':['C'] * 10,
                        'solvothermal': {
                            "results": [
                                {
                                    'reactant':{
                                        'coordinates': np.random.rand(10, 3),
                                        'hessian': np.random.rand(30, 30),
                                    },

                                    'transition_state': {
                                        'coordinates': np.random.rand(10, 3),
                                        'hessian': np.random.rand(30, 30)
                                    }
                                }

                            ]
                        },

                        'internal_0': {
                            'spec':[1, 3, 2],
                            "results": [
                                {
                                    'reactant': {
                                        'coordinates': np.random.rand(10, 3)
                                    },

                                    'transition_state': {
                                        'coordinates': np.random.rand(10, 3)
                                    }
                                }

                            ]
                        }
                    }
                }
            }
        }

        ugh = schema.compress_tree(data)

        # huh = schema.validate_distortion_data(data)

        with tf.TemporaryDirectory() as td:
            js_file = os.path.join(td, 'tree.json')
            schema.write_distortion_data(js_file, data, mode='json')
            new_data = schema.read_distortion_data(js_file, data, mode='json')

            np_file = os.path.join(td, 'tree.npz')
            schema.write_distortion_data(np_file, data)
            new_data = schema.read_distortion_data(np_file, data)

    def test_CompileReactionSets(self):
        thc_res_loc = os.path.expanduser('~/Documents/Postdoc/Projects/CMCC/DA_res')
        os.chdir(thc_res_loc)
        for k in range(4, 6):
            comp = thc_tools.compile_reaction_class(f'Results_{k+1}')
            thc_tools.write_aggregate_data(f'Res{k+1}_aggregate.npz', comp)
            remp = thc_tools.load_aggregate_data(f'Res{k+1}_aggregate.npz')

        # print(comp.keys())
        # print(comp['coords'][0].shape)
        # for r,c in zip(remp['coords'], comp['coords']):
        #     print(r-c)


if __name__ == '__main__':
    os.chdir(root)
    unittest.main('tests.CMCCTests')