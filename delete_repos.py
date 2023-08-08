from benchmark_qa import all_augment_cols
from huggingface_hub import delete_repo
from tqdm import tqdm
from huggingface_hub import login

username = "parinzee"

if __name__ == "__main__":
    # Confirm
    print("Are you sure you want to delete all the repos from claq?")
    print("This action cannot be undone.")
    print("Type 'yes' to confirm.")
    confirm = input("> ")

    if confirm.lower() != "yes":
        print("Aborting.")
        exit()

    for col in tqdm(all_augment_cols):
        try:
            if col:
                for ratio in range(1, 11):
                    ratio = ratio / 10

                    print(f"Deleting {username}/claq-qa-th-wangchanberta-{col}_{ratio}")
                    delete_repo(f"{username}/claq-qa-th-wangchanberta-{col}_{ratio}")
            else:
                print(f"Deleting {username}/claq-qa-th-wangchanberta-original")
                delete_repo(f"{username}/claq-qa-th-wangchanberta-original")
        except Exception as e:
            print(e)
            print("Failed to delete repo.")
            continue