import os
from pdf2image import convert_from_path
import PIL
#4896*6336
pdf_dir = r"C:\Users\randy\Downloads\the_way_to_train"
save_dir = r'C:\Users\randy\Downloads\the_way_to_train\PDF2IMG'

os.chdir(pdf_dir)
for pdf_file in os.listdir(pdf_dir):

    if pdf_file.endswith(".pdf"):

        pages = convert_from_path(pdf_file, dpi=96)
        pdf_file = pdf_file[:-4]

        for page in pages:
            print("%s/%s-page%d.png" % (save_dir,pdf_file,pages.index(page)))
           # page.save("%s/%s-page%d.png" % (save_dir,pdf_file,pages.index(page)), "png")
            if page == [-1]:
                print(page)

'''
if __name__ == "__main__":
    pdf2img(pdf_dir)
'''
