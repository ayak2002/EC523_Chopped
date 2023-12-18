import requests
from PIL import Image
from io import BytesIO
import os

# List of image URLs
image_urls = [
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQx-YXBorZO-lfH-5FhP8lUdhzANQKgvR67rOttXwjn0ZWnEs8Y&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQ5xshyalxjsK1FfhzPmdXx3jxsAtJvfh5i58tTrfioRgoQbkzx&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSBzAH9cULvAM7CGM_lUGeYeeBRniHolyJRi3I2dC5yGn_xBD9g&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQX5Qb3gKXSnJrmpfCcLSLedXHVz6NmAR99cvnEYgTzzt9tsGNj&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcT8nMUy15XhEHJfX5uENnoKsL9C1LnIoFQxZydYzum-TqvRRnn6&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRjjniHqz3VkKZ6xYGY6Krvt6s5ZC3iKBbzQkW5d36KIz6PhePc&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTf0f_SQG-NYzlpmvw7-pyCmwhMkxprUcUZ5X37RcjKzWg8nz7X&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRdwwfDqPbahRZGwtdyyxyZu9uKG_5WEwc815Q6v4OdPDrjtx8o&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTJtKyi-wuxKL9BgTMJufhTIUMrcFRsA52DRxxjOUDjTsdMVun-&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTfGzHAZVd2uJhoXNqqFbAG_CblsWv4l76yQQ1qlPikWarXOYx6&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRvN6jMG52JHjF3NzkRMO_7YXmDMa4aqyWhs8Cj5kp6IScVyBqh&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcR5lcf-WFtPhBz8gRiTj0R8q2W16bRX49QL4N5Kll5WSWtWyfeE&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTaWLAU3msH4X7RTU92m7F9BlKcPGsOQRfpYMGuIPVMdB9NfBQB&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSFMe9928cjtVMs-IP4EQ3BtwBeiBj7eUS4EgcpcnjppZb5YRtO&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRKjQfNuLw7sK6cpQhalXlqHYJFyBAihAfpnPFplyYba4UKwJLJ&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRjWkeFpoV1DU7QoK9nTDi63kHfnoVxAmTGl5OiSuPMC7jcH_Cm&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSHkvUYKSdjNs1KMOHsSqv7cshkToYJ36SZtWaqCqshk_fnpDRB&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTEXn4oytnxV4CXJCbqqn9_w6V6jkvw_EH4amHheemVav-eMjmS&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRxrYY-DrEMTQWLu_lvJLph9Cb7mFVl0QVY8nrD63zyI8WY5PlH&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTktkH0DZWo9mci7WPqyIRjEDXCiVwoXFzDmMKb1Lp7c-SdAsX0&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSHUWcRrqxsxCyyPifx1tN8w0rwHeM0cPZdjPxQ6Z8QNioGQRge&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTnXKmIdQcgyqkIBgbolqrFf6dH3uiPP1rfofqiTo65qaKQImSZ&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQUQsWc3N2FAe-BNxQEu1HsHXmTuvUDcHtnQGyK20-Lr1VuIbdx&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcR_y5KaaiHOsZQ_BAh4j6_uWkL1qLoxGxI1M0DRRlf9iwCsjhcJ&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSqb4y35tTexMR6X9r3zQWwPvLuzcpCoPfeX4OQtgJs3gSnuM41&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRbmSIZBXrKDRAvO5loH6E0kZ80XBOHcPAOpisfFT3sJ6dTNkel&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRHnGjB0-T4YBy_YoNIx2pvfQ4oCYHnVFEAL25utiXuG7WvWk_f&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQ91nEGz4Dvr4-lNQKKKYrt7E-iiJWBl5OMkcjM8LhnqD-SZqWe&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcS3CIJLQOA19VvVj1CvV3Bnus1U_mioV5OPhu91T9WvMtfSRY7C&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSpGjbv2c5RtNfvBu0wjS0BmREHDdOwHqCsa-rX7ginZZEm1Lvg&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcS8ULMz0bA0oYmWyDZVSvzPc5h2pRx0A60cT2EDPdfp6AWfkMUN&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQLFIqRzJZwQCutwdmZdJkDNa974n_zruhYwN-KvBP-jW29hgKp&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQsBUmIijlpPVYSizyK5urmucyUZPXtFtoG0sQs63gQxHFX0eRc&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQt4sUJjtsEM1kA3_ChXTdtb15ndm--andew-vrn_LCzv_l_o8Y&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQCz24-el9yFHl9w2cLAE6I2Xe-tKWDaR6Twd6pTDBGc5_ksYV2&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQppT6KoprqJNAI5tiKEMSU40VJqp9jzGRt2rj4o_2_fpzgtHjR&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTpS_WmGZwjEghBegw8WS-1RyNgoMgJfxr0RbXdpMoEPHMh1mah&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTW_Q662GvR0zGq6eGOWqtjId6ulYCR_h2bSmOgIgBiVxuV2xs0&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTMPXz3ahfnh64vF3dfOCKAoBN-h4BvfXA_-oKoF270k8mD5suZ&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQpy_0LCJOUOYMeNBxqF2-McJVx3JgS1BLrjpvCiZ-5GaPJqnnh&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRecY7BEi8G5o2kHLYdhD8gjNGHqUJsCPcy0FnVohaxxvMBS6BR&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSVF5YLcPkGiZZxEuHRZGoqj4VNixVIaj2crYy_QxmWtv7mPVk0&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRv7BUdd2QkUs7ncP67UiKuJN4oBiRtAt9kkMVzifIynVQ8lbt7&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQuoYyHaFSB-cDib_ODxH6Nmp27Hjxfn8y0ruLeuBMOUoRWDY1g&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTL8XP6uQGAh37ItqTy70fuT-lqWH9_OJYJtO-Y9D9AwvqKD9do&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQ2VESTB_iR8ylwusDG6DMD0KJUgjDibEzKy1C5Fds6NKcn92Nb&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTnXaoIhhsaToyzcll0sDr7qPdhICIC5FsuVkE2q_HbWAj-jF73&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTlslwkFny7MAGhqE_Zx3-VnF-kheq-AU5h6yKsziX_2rGAnXlp&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQ87Bmc1GLqz9m3YIyjlWxDwvIOG6TOVI6YQ1nKD4IevVWF1X6e&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSKA2IwCmbFH-3BV6lH0Y2hNWgcE2Enss8J3MVPpGP4s4rnfUSq&usqp=CAU',
'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSayPsmK_aGBdimikwhACbEjTyG3s7_wWLpp2yUprbx0CqN9TbX&usqp=CAU'
]




# Folder to save the images
save_folder = "filipino_food_data/test/ube_ice_cream"
os.makedirs(save_folder, exist_ok=True)

# Function to download and save an image
def download_image(url, folder, index):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error if the request failed
        img = Image.open(BytesIO(response.content))
        img.save(os.path.join(folder, f"image_{index}.jpg"))
        print("downloaded")
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")

# Downloading each image
for i, url in enumerate(image_urls):
    download_image(url, save_folder, i)
