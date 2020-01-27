from auvlib.bathy_maps import map_draper
from matplotlib import pyplot as plt

#Load sss_map_images
images = map_draper.sss_map_image.read_data("/home/auv/Scripts/sss_map_images_corrected_SSH1.cereal")

#Show map_image and waterfall_image
counter = 0
for image in images:
    counter += 1
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(image.sss_map_image)
    ax2.imshow(image.sss_waterfall_image)
    plt.savefig("Figure"+str(counter))
