from core_utils import InitializeData, GeneratePlots


if __name__ == "__main__":
    meta = "../data/ClasesImagenes.csv"
    data = "../data/CarpetaImagenes/"
    i = InitializeData(meta, data)
    p = GeneratePlots(i.df, i.image_data)
    p.generate_histogram(1)
