from core_utils import InitializeData, GeneratePlots


if __name__ == "__main__":
    meta = "../data/ClasesImagenes.csv"
    data = "../data/CarpetaImagenes/"
    PATH_DATA = "kurtosis.csv"
    PATH_IMAGE = "kurtosis_boxplot.eps"

    i = InitializeData(meta, data)
    p = GeneratePlots(i.df, i.image_data)
    p.generate_boxplot_kurtosis(PATH_DATA, PATH_IMAGE)
