import os, sys

import numpy as np

dev_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))
root = os.path.join(dev_root, 'cmcc-reactions')
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
# import cmcc_reactions.data_analysis_tools as thc_tools
import cmcc_reactions.generate_reaction_products as gen_prods
import cmcc_reactions.reaction_data_analysis as rda
import cmcc_reactions.coordinate_choice as cocho
import cmcc_reactions.utils as utils

__all__ = [
    "CMCCTests"
]


def test_data(*path):
    return os.path.join(root, "tests", "TestData", *path)
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

    @unittest.skip
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

    @unittest.skip
    def test_GenerateReactionProducts(self):
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        dienes = [
            ["CH", "CH", "CH", "CH"],
            # ["C@@H", "C", "C@H", "C@H"],
            # ["C@@H", "C", "C@H", "C@@H"],
            # ["C@H", "C", "C@@H", "C@H"]
        ]
        diene_template = "[{D[0]}:1]([CH2:7]2)[{D[1]}:5]=[{D[2]}:6][{D[3]}:2]2"
        base_templates = [
            "[R][C@@H:3]1[diene][C@@H:4]1[X]",
            "[R][C@@H:3]1[diene][C@H:4]1[X]"
        ]
        replacements = {
            'R': ["O=S(=O)(C2=CC=CC=C2)", "NC", "NC(=O)", "FC(C=C3)=CC=C3"],
            'X': ["S(=O)(C4=CC=CC=C4)=O", "CN", "C(=O)N", "C5=CC=C(F)C=C5"],
            "R'": ["S(=O)(C6=CC=CC=C6)=O", "CN", "C(=O)N", "C7=CC=C(F)C=C7"]
        }

        import shutil
        output_dir = os.path.expanduser('~/Desktop/test_smi')
        shutil.rmtree(output_dir)
        dat = gen_prods.generate_products_and_optimize(
            base_templates[:1],
            dienes,
            {
                k: v
                for k, v in replacements.items()
            },
            diene_template=diene_template,
            evaluate_energy=False,
            preoptimize=False,
            output_dir=output_dir,
            parallelizer=None,
            verbose=True
        )
        print(dat)

        # mod_smi = gen_prods.modify_template(gen_prods.template_1, {'R':'CCC', 'X':'ONO', "R'":'F'})
        # mod_smi = gen_prods.modify_template(gen_prods.template_5, {'R':'CCC', 'X':'ONO', "R'":'F'})
        # print(
        #     mod_smi,
        #     gen_prods.bond_breaking_indices(mod_smi)
        # )

    @unittest.skip
    def test_InitialSampling(self):
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        init_data = utils.read_namedtuple(
            test_data('problem_product.json'),
            nt_type='InitialProductData'
        )

        presamp = gen_prods.generate_reactants_from_products(
            init_data,
            max_iterations=0,
            # output_dir=test_data()
        )

        return

        init_data = utils.read_namedtuple(
            test_data('product.json'),
            nt_type='InitialProductData'
        )

        presamp = gen_prods.generate_reactants_from_products(
            init_data,
            max_iterations=5,
            output_dir=test_data()
        )

    @unittest.skip
    def test_RefinedSampling(self):
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        init_data = utils.read_namedtuple(
            test_data('product_tris.json'),
            nt_type='InitialProductData'
        )
        traj_data = utils.read_namedtuple(
            test_data('trajectory_tris.json'),
            nt_type='ReoptimizedTrajectoryData'
        )

        new_traj = gen_prods.refine_trajectory(
            init_data,
            traj_data
        )

        import McUtils.Plots as plt
        plt.Plot(
            new_traj.final_rmsds,
            new_traj.final_energies
        ).show()

    @unittest.skip
    def test_CoordinateSystemGen(self):
        from Psience.Molecools import Molecule

        ts_samp = Molecule.from_file(test_data('ts_samp.xyz'))
        cocho.get_mostly_fixed_coordinate_system(ts_samp, [(22, 18, 19, 20)])

    @unittest.skip
    def test_CompressFSTree(self):
        tree = utils.construct_json_file_tree(test_data('test_smi'))
        import pprint
        pprint.pprint(tree)

    def test_DA_Analysis(self):
        rda.DielsAlderReactionTrajectory.from_file(
            test_data('refined.json')
        ).plot_profile(
            distance_metric=rda.incremental_rmsds
        ).show()


if __name__ == '__main__':
    os.chdir(root)
    unittest.main('tests.CMCCTests')