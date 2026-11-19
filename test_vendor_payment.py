
import os

folder_path = "documents"

for file_name in os.listdir(folder_path):

    if file_name.startswith("VND"):

        file_path = os.path.join(folder_path, file_name)

        print("\n" + "=" * 80)
        print("FILE:", file_name)
        print("=" * 80)

        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        print(content)
        print("\n" + "=" * 80)
