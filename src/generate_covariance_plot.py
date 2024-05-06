from core_utils import InitializeData, GeneratePlots


if __name__ == "__main__":
    meta = "../data/ClasesImagenes.csv"
    data = "../data/CarpetaImagenes/"

    i = InitializeData(meta, data)

    PATH_IMAGE = "pca_covariance.eps"
    ZOOM = None

    p = GeneratePlots(i.df, i.image_data)
    p.generate_covariance_plot(PATH_IMAGE, ZOOM)


    PATH_IMAGE = "pca_covariance_zoomed.eps"
    X_LIM = [0, 2000]
    Y_LIM = [90, 100]
    ZOOM = [X_LIM, Y_LIM]

    p = GeneratePlots(i.df, i.image_data)
    p.generate_covariance_plot(PATH_IMAGE, ZOOM)
