from utils import load_data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from colorama import init, Fore, Style
from tqdm import tqdm
import numpy as np

init(autoreset=True)

def highlight(text, color=Fore.CYAN):
    return f"{color}{text}{Style.RESET_ALL}"

print(highlight("🚀 Loading data...", Fore.YELLOW))
data, labels = load_data("train/PetImages", max_images=300)
print(highlight("✅ Data Loaded Successfully!", Fore.GREEN))
print(highlight(f"📊 Total images loaded: {len(data)}", Fore.CYAN))

print(highlight("✂️ Splitting data...", Fore.YELLOW))
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(highlight("🧠 Training model...", Fore.YELLOW))
model = SVC(kernel='linear')
model.fit(X_train, y_train)

print(highlight("📈 Evaluating model...\n", Fore.YELLOW))
predictions = model.predict(X_test)

print(highlight("🎯 Model Performance:\n", Fore.MAGENTA))
print(highlight(classification_report(y_test, predictions), Fore.WHITE))

print(highlight("🎉 Execution Completed!", Fore.GREEN))
print(highlight("📌 Summary:", Fore.CYAN))
print(f"""
- Total Images Used  : {len(data)}
- Train/Test Split   : {len(X_train)}/{len(X_test)}
- Model Used         : SVM (Support Vector Machine)
- Accuracy Achieved  : {accuracy_score(y_test, predictions):.2f}
""")
