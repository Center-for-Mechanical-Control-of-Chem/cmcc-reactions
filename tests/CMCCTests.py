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
import cmcc_reactions.trajectory_tools as trajt
import cmcc_reactions.reaction_data_analysis as rda
import cmcc_reactions.utils as utils
import cmcc_reactions.pipeline as pipeline
import cmcc_reactions.generate_reaction_products as gen_prods
import cmcc_reactions.coordinate_choice as cocho
import cmcc_reactions.optimal_directions as fopt

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

        # init_data = utils.read_namedtuple(
        #     test_data('trajectory.json'),
        #     # nt_type='InitialProductData'
        # )
        traj_data = utils.read_namedtuple(
            test_data('trajectory_problem.json'),
            # nt_type='ReoptimizedTrajectoryData'
        )
        traj_data = utils.read_namedtuple(
            test_data('trajectory.json'),
            # nt_type='ReoptimizedTrajectoryData'
        )

        new_traj = gen_prods.refine_trajectory(
            traj_data,
            profile_generator='pys-gsm',
            coord_type='cartesian',
            # num_images=30,
            # max_iterations=5,
            # param='energy',
            # optimizer='lbfgs'
            # ts_opt_generator=None
            # ts_opt_settings={}
        )

        rda.compare_profiles(new_traj, marker='.').show()

    @unittest.skip
    def test_LocalizedOptimization(self):
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        traj_file = test_data('profile_problem.json')
        base_file = test_data('trajectory_problem.json')
        reoptimize = False
        use_original = False
        if (reoptimize and use_original) or not os.path.exists(traj_file):
            pre_string = utils.read_namedtuple(base_file)
            new_traj = gen_prods.refine_trajectory(pre_string,
                                                   refine_endpoints=True,
                                                   optimizer_settings=dict(thresh='gau_tight'))
            # new_traj = gen_prods.refine_trajectory(pre_string)
            ref1 = rda.DielsAlderReactionTrajectory.from_trajectory_data(new_traj)
            ref1.save(traj_file)
        elif reoptimize:
            pre_string = utils.read_namedtuple(traj_file)
            new_traj = gen_prods.refine_trajectory(pre_string,
                                                   refine_endpoints=True,
                                                   optimizer_settings=dict(thresh='gau_tight'))
            # new_traj = gen_prods.refine_trajectory(pre_string)
            ref1 = rda.DielsAlderReactionTrajectory.from_trajectory_data(new_traj)
            ref1.save(traj_file)
        else:
            ref1 = rda.DielsAlderReactionTrajectory.from_file(traj_file)
        opt = fopt.ForceOptimizer(ref1.reactant, ref1.transition_state, fragment_indices=1)
        opt.animate_normed(0, backend='x3d', mag=2).show()

    @unittest.skip
    def test_AdjustedForceOptimizer(self):
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        import McUtils.Coordinerds as coordops

        ref1 = rda.DielsAlderReactionTrajectory.from_file(test_data('trajectory_problem.json'))

        zm_coords = [
            coordops.extract_zmatrix_internals(z)
            for z in ref1.reactant.get_bond_zmatrix(connect_fragments=False,
                                                    fragment_ordering=[0, 1],
                                                    attachment_points={0: 2}
                                                    )
        ]
        all_ints = sum(zm_coords, [])
        zm_full = ref1.reactant.get_bond_zmatrix(
            connect_fragments=True,
            fragment_ordering=[0, 1],
            attachment_points={0: 2}
        )

        opt = fopt.ForceOptimizer(ref1.reactant, ref1.transition_state,
                                  # fragment_indices=list(range(len(ref1.reactant.atoms)))[4:],
                                  # fragment_indices=1,
                                  fragment_indices=ref1.reactant.fragment_indices[1][3:],
                                  remove_fragment_transrot=False,
                                  remove_local_transrot=True,
                                  allow_mode_mixing=True,
                                  projection_internals=zm_coords[1],
                                  internals=zm_full
                                  # project_internals=False,
                                  # use_mode_space=False
                                  )

        opt.save(test_data('optimized_forces.json'))

    @unittest.skip
    def test_ForceAdjustedProfile(self):
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        from McUtils.Data import UnitsData

        opt = fopt.ForceOptimizer.from_file(test_data('optimized_forces.json'))

        r, ts, fmrd = opt.reoptimize_with_force(0,
                                                -100,
                                                units='PicoJoules/Meters',
                                                # units=None,
                                                # optimizer_mode='pysis',
                                                # optimizer_method='lbfgs',
                                                # profile_generator='pys-dimer',
                                                # profile_generator='relaxed',
                                                # profile_generator='pys-cos',
                                                # optimizer_mode='ase',
                                                # optimizer_method='bfgs',
                                                profile_generator='ase-dimer',
                                                # apply_constraints=False,
                                                # modify_forces=True,
                                                # optimizer_mode='default',
                                                # optimizer_method='quasi-newton',
                                                optimizer_mode='scipy',
                                                optimizer_method='bfgs',
                                                apply_constraints=True,
                                                use_internals=False,
                                                mass_weight=False,
                                                reoptimize_reactants=True,
                                                reoptimize_ts=False,
                                                initial_ts_step=1,
                                                initial_reactants_step=1,
                                                # num_ts_steps=3,
                                                max_iterations=1000,
                                                max_displacement=.05,
                                                track_best=False)
        fmrd: fopt.ForceModifiedReactionData

        rf, tf, r0, t0 = [
            e * UnitsData.convert("Hartrees", "Kilocalories/Mole")
            for e in [
                fmrd.force_modified_reactant_energy,
                fmrd.force_modified_transition_state_energy,
                fmrd.reactant_energy,
                fmrd.transition_state_energy
            ]
        ]
        # print([
        #     rf, tf, r0, t0,
        # ])
        print(rf - r0)
        print(tf - t0)

        scan = r.embed_coords(
            [fmrd.reactant_geom, fmrd.force_modified_reactant_geom],
            sel=r.fragment_indices[0]
        )
        r.plot(scan, include_save_buttons=True, background='white').show()

        scan = ts.embed_coords(
            [fmrd.transition_state_geom, fmrd.force_modified_transition_state_geom],
            sel=r.fragment_indices[0]
        )
        ts.plot(scan, include_save_buttons=True, background='white').show()

    @unittest.skip
    def test_ForceOptimizerPipeline(self):
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        uuh = pipeline.run_optimization_pipeline(
            test_data('product.json'),
            output_file=test_data('pipline_output.json'),
            max_iterations=5
        )
        print(uuh)

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

    @unittest.skip
    def test_DA_Analysis(self):
        rda.DielsAlderReactionTrajectory.from_file(
            test_data('refined.json')
        ).plot_profile(
            distance_metric=rda.incremental_rmsds
        ).show()

    @unittest.skip
    def test_NewBuildingBlocks(self):
        # from Psience.Molecools import Molecule

        base_template = '[CH2:3]1[CH:1]([CH2:7]2)[CH:5]=[CH:6][CH:2]2[CH2:4]1'

        temp = base_template
        # Molecule.from_string(base_template).plot().show()
        # Molecule.from_string("[CH3:1]N").plot().show()

        temp = gen_prods.join_fragments(temp,
                                        "[CH3:1]N",
                                        [[2, 0]]
                                        )
        temp = gen_prods.join_fragments(temp,
                                        "[SH:1](=O)(C4=CC=CC=C4)=O",
                                        [[3, 0]]
                                        )
        temp = gen_prods.set_chiralities(temp, {3:'cw', 4:'cw'})
        print(temp)
        # Molecule.from_string(temp).plot().show()
        # print(
        #     gen_prods.join_fragments(
        #         '[C@@H:3]1[CH:1]([CH2:7]2)[CH:5]=[CH:6][CH:2]2[C@@H:4]1',
        #         "N[C:1]",
        #         [[2, 0]]
        #     )
        # )

    @unittest.skip
    def test_NewEnumeration(self):
        from Psience.Molecools import Molecule

        for n,smi in enumerate(gen_prods.fragment_to_smiles_iterator(
            '[C:3]1[C:1]([C:7]2)[C:5]=[C:6][C:2]2[C:4]1',
            ["[C:1]N", "[S:1](=O)(C4=CC=CC=C4)=O"],
            [2, 3],
            chiralities=[['cw', 'ccw'], ['cw', 'ccw']]
        )):
            Molecule.from_string(smi).plot(highlight_atoms=[0, 1]).show()
            if n >= 4:
                break

    @unittest.skip
    def test_ReoptZMIssues(self):
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # raise Exception(
        #     gen_prods.inchi_key('N[CH2:8][C@H:3]1[CH:1]2[CH:5]=[CH:6][C@@H:2]([CH:4]1[S:9](=O)(=O)c1ccccc1)[CH2:7]2')
        # )

        uuh = pipeline.run_optimization_pipeline(
            test_data('problem_product2.json'),
            output_file=test_data('problem_pipline_output2.json'),
            max_iterations=5
        )
        print(uuh)

    @unittest.skip
    def test_ResultsAnalysis(self):
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        res = pipeline.OptimizedForceResults.from_file(test_data('pipeline_data3.json'))
        # res = pipeline.run_optimization_pipeline(
        #     test_data('pipeline_data_new.json'),
        #     test_data('pipeline_data_new.json'),
        #     steps=['fmrds'],
        #     verbose=True,
        #     # max_iterations=5
        # )

        o = res.optimizer
        r, t = o.rs, o.ts
        uuh = fopt.ForceOptimizer(r, t, internals=o.internals)
        print(o.force_coeffs.shape,
              uuh.force_coeffs.shape,
              uuh.to_data().force_coeffs.shape)
        uuh2 = fopt.ForceOptimizer.from_data(uuh.to_data())
        print(uuh.force_coeffs.shape,
              uuh2.force_coeffs.shape)

    @unittest.skip
    def test_Refinements(self):
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        traj = utils.read_namedtuple(test_data('reopt_test.json'))
        yeesh = gen_prods.refine_trajectory(
            traj,
            refine_endpoints=False,
            # num_images=6,
            # profile_generator=None,
            profile_generator='pys-ts',
            # profile_generator='pys-cos',
            # ts_opt_generator='pys-ts',
            # spring_constant=.01,
            # max_iterations=50,
            max_refinement_iterations=50,
            logger=True,
            # coord_type='dlc'
            optimizer_settings=dict(coord_diff_thresh=0.0005)
        )
        trajt.compare_profiles(yeesh, marker='o').show()

    @unittest.skip
    def test_InternalsForces(self):
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        traj = pipeline.OptimizedForceResults.from_file(test_data('pipeline_data.json'))

        import subprocess
        import memray
        import os
        try:
            os.remove(os.path.expanduser("~/Desktop/memprof.out"))
        except:
            ...
        with memray.Tracker("/Users/Mark/Desktop/memprof.out"):
            opt = traj.optimizer.reoptimize_internals_with_force(
                'dihedrals',
                max_internals=2,
                max_iterations=100
            )

        try:
            os.remove(os.path.expanduser("~/Desktop/memprof_graph.html"))
        except:
            ...
        subprocess.run(["memray", "flamegraph", os.path.expanduser("~/Desktop/memprof.out"), "-o", os.path.expanduser("~/Desktop/memprof_graph.html")])
        subprocess.run(["open",  os.path.expanduser("~/Desktop/memprof_graph.html")])

    @unittest.skip
    def test_RandomForces(self):
        import McUtils.Numputils as nput

        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        traj = pipeline.OptimizedForceResults.from_file(test_data('pipeline_data.json'))
        optimizer = traj.optimizer
        dir2 = nput.find_basis(
            np.random.normal(size=(10, 3 * len(optimizer.ts.atoms))).T,
            method='qr'
        ).T
        opt = traj.optimizer.reoptimize_with_force(
            [0, 1],
            displacements=dir2,
            max_iterations=5
        )

    @unittest.skip
    def test_HydrostaticForces(self):

        import McUtils.Numputils as nput
        from McUtils.Data import UnitsData
        from Psience.Molecools import Molecule

        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        traj = pipeline.OptimizedForceResults.from_file(test_data('pipeline_data.json'))
        # optimizer = traj.optimizer
        # dir2 = nput.find_basis(
        #     np.random.normal(size=(10, 3 * len(optimizer.ts.atoms))).T,
        #     method='qr'
        # ).T


        # def apply_hydrostatic(ts:Molecule, coords, surface_points=500, radius_scaling=1.2):
        #     coords = np.asanyarray(coords).reshape((-1, 3))
        #     surf = ts.modify(coords=coords).get_surface(samples=surface_points, radius_scaling=radius_scaling).get_triangulation()
        #     groups, _ = nput.group_by(np.arange(len(surf.tri_map)), surf.tri_map)
        #     areas = surf.surface_area(return_components=True)
        #     area_fractions = areas / np.sum(areas)
        #     scaled_normals = area_fractions[..., np.newaxis] * surf.normals
        #     force = np.zeros_like(coords)
        #     for atom, inds in zip(*groups):
        #         force[atom] = np.sum(scaled_normals[inds,], axis=0)
        #     return force.flatten()[np.newaxis]
        #
        #     # surf
        #     # compute terms normal to the surface
        #     ...

        # fig = traj.optimizer.rs.plot(backend='x3d')
        # surf = traj.optimizer.rs.get_surface(samples=500, radius_scaling=1.2).get_triangulation()
        # surf.plot(figure=fig)
        # fig.show()

        _, _, opt = traj.optimizer.reoptimize_with_pressure(5,
                                                            "Gigapascals",
                                                            pressure_model='cylinder',
                                                            pressure_options={
                                                                'axis':'-c',
                                                                'radius':1 * UnitsData.convert("Angstroms", "BohrRadius"),
                                                                'bidirectional':True
                                                            },
                                                            max_iterations=50,
                                                            logger=True)

        print(
            (opt.transition_state_energy - opt.reactant_energy) * UnitsData.convert("Hartrees", "Kilocalories/Mole"),
            (opt.force_modified_transition_state_energy - opt.force_modified_reactant_energy)* UnitsData.convert("Hartrees", "Kilocalories/Mole")
        )

        traj.plot_profile(fmrd=opt, force_modified='both').show()

        traj.optimizer.rs.plot([
            opt.reactant_geom,
            opt.force_modified_reactant_geom
        ]).show()

    @unittest.skip
    def test_ExportNew(self):
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        res = pipeline.OptimizedForceResults.from_file(test_data('pipeline_data.json'))

        npdat = utils.dumps_namedtuple(res.to_data(), mode='npz')
        print(utils.loads_namedtuple(npdat, mode='npz', decompress=True))

        res.compare_profiles().show()

    def test_LoadPipelineNPZ(self):
        uuh = pipeline.read_compressed_pipeline_data('/Users/Mark/Documents/Postdoc/Projects/CMCC/disub_alt.npz')

if __name__ == '__main__':
    os.chdir(root)
    unittest.main('tests.CMCCTests')