import os

#Project directories

root_path = 'C:\\Users\\Sanchit Kumar\\PycharmProjects\\SOEN_691_Project\\'
data_folder = os.path.join(root_path , 'Data')
src_folder = os.path.join(root_path , 'Src')


#Dataset used in the project
data_file_name  = 'fake_or_real_news.csv'
data_file_path = os.path.join(data_folder , data_file_name)