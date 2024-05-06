from core_utils import Generate2dVisualization


if __name__ == "__main__":
    meta = "../data/ClasesImagenes.csv"
    data = "../data/CarpetaImagenes/"

    g = Generate2dVisualization(meta, data)

    PCA_VARIANT = 'linear'
    KERNEL = None
    PATH_DATA = 'linear_pca_scatter.csv'
    PATH_IMAGE = 'linear_pca_scatter.eps'
    g.generate_pca(PCA_VARIANT, KERNEL, PATH_DATA, PATH_IMAGE)

    PCA_VARIANT = 'kernel'
    KERNEL = 'poly'
    PATH_DATA = 'kpca_poly_scatter.csv'
    PATH_IMAGE = 'kpca_poly_scatter.eps'
    g.generate_pca(PCA_VARIANT, KERNEL, PATH_DATA, PATH_IMAGE)

    PCA_VARIANT = 'kernel'
    KERNEL = 'rbf'
    PATH_DATA = 'kpca_rbf_scatter.csv'
    PATH_IMAGE = 'kpca_rbf_scatter.eps'
    g.generate_pca(PCA_VARIANT, KERNEL, PATH_DATA, PATH_IMAGE)

    PCA_VARIANT = 'kernel'
    KERNEL = 'sigmoid'
    PATH_DATA = 'kpca_sigmoid_scatter.csv'
    PATH_IMAGE = 'kpca_sigmoid_scatter.eps'
    g.generate_pca(PCA_VARIANT, KERNEL, PATH_DATA, PATH_IMAGE)
