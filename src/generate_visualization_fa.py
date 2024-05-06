from core_utils import Generate2dVisualization


if __name__ == "__main__":
    meta = "../data/ClasesImagenes.csv"
    data = "../data/CarpetaImagenes/"

    g = Generate2dVisualization(meta, data)

    PATH_DATA = 'fa_randomized_none.csv'
    PATH_IMAGE = 'fa_randomized_none.eps'
    SVD = 'randomized'
    ROTATION = None
    g.generate_fa(PATH_DATA, PATH_IMAGE, SVD, ROTATION)

    PATH_DATA = 'fa_randomized_varimax.csv'
    PATH_IMAGE = 'fa_randomized_varimax.eps'
    SVD = 'randomized'
    ROTATION = 'varimax'
    g.generate_fa(PATH_DATA, PATH_IMAGE, SVD, ROTATION)

    PATH_DATA = 'fa_lapack_varimax.csv'
    PATH_IMAGE = 'fa_lapack_varimax.eps'
    SVD = 'lapack'
    ROTATION = 'varimax'
    g.generate_fa(PATH_DATA, PATH_IMAGE, SVD, ROTATION)
