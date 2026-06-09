from cmcc_reactions.pipeline import generate_from_product_library_set, BASE_TEMPLATES, BASE_FRAGMENTS

TEMPLATES = [
    'cyclopentadiene',
    'butadiene',
]
FRAGMENT_LIST = ["OMe", "pF-phenyl"]
ACTIVE_SITES = [
    [2],
    [2, 3],
    [2, 2, 3],
    [2, 2, 3, 3]
]

product_libraries = [
    dict(
        template=BASE_TEMPLATES[temp],
        fragments=[BASE_FRAGMENTS[f] for f in FRAGMENT_LIST],
        active_sites=active_sites,
        chiralities=[['cw', 'ccw'], ['cw', 'ccw']]
    )
    for temp in TEMPLATES
    for active_sites in ACTIVE_SITES
]

generate_from_product_library_set(
    product_libraries,
    output_dir='product_library'
)
