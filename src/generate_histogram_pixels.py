from core_utils import InitializeData, GeneratePlots


if __name__ == "__main__":
    meta = "../data/ClasesImagenes.csv"
    data = "../data/CarpetaImagenes/"
    i = InitializeData(meta, data)

    IMAGE_NUMBER = 1
    PATH_IMAGE = 'pixel_histogram_cancerous.eps'
    p = GeneratePlots(i.df, i.image_data)
    p.generate_histogram(IMAGE_NUMBER, PATH_IMAGE)

    IMAGE_NUMBER = 0
    PATH_IMAGE = 'pixel_histogram_healthy.eps'
    p = GeneratePlots(i.df, i.image_data)
    p.generate_histogram(IMAGE_NUMBER, PATH_IMAGE)
