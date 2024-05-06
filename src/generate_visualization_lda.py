from core_utils import Generate2dVisualization


if __name__ == "__main__":
    meta = "../data/ClasesImagenes.csv"
    data = "../data/CarpetaImagenes/"

    g = Generate2dVisualization(meta, data)

    PATH_DATA = 'lda_scatter.csv'
    PATH_IMAGE = 'lda_scatter.eps'
    g.generate_lda(PATH_DATA, PATH_IMAGE)
