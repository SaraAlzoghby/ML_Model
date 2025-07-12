import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans  # Added for clustering


def preprocess_data(controller):
    df = controller.df.copy()
    df.dropna(inplace=True)
    if df.empty:
        raise ValueError("Dataset is empty after removing missing values")
    for col in df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        controller.label_encoders[col] = le
    controller.df = df

def train_model(controller):
    df = controller.df
    target_col = controller.target_column
    
    task = controller.selected_task
    
    if task == "clustering":
        # For clustering, we don't need a target column
        X = df
        controller.scaler = StandardScaler()
        X = pd.DataFrame(controller.scaler.fit_transform(X), columns=X.columns, index=X.index)
        
        algo = controller.selected_algo
        if algo == "K-Means":
            model = KMeans(n_clusters=3, random_state=42)
            model.fit(X)
            controller.model = model
            clusters = model.predict(X)
            controller.y_pred = clusters  # Store cluster assignments
            
            # Return cluster metrics
            return f"K-Means Clustering Results:\nNumber of clusters: 3\nInertia: {model.inertia_:.2f}", clusters
    else:
        # Existing classification/regression code
        X = df.drop(columns=[target_col])
        y = df[target_col]

        if task == "classification":
            if not np.issubdtype(y.dtype, np.integer) or len(y.unique()) > 10:
                raise ValueError("Classification requires a discrete target variable with reasonable number of classes")
        else:
            if not np.issubdtype(y.dtype, np.number):
                raise ValueError("Regression requires a numeric target variable")

        controller.scaler = StandardScaler()
        X = pd.DataFrame(controller.scaler.fit_transform(X), columns=X.columns, index=X.index)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        controller.X_train = X_train
        controller.X_test = X_test
        controller.y_train = y_train
        controller.y_test = y_test

        algo = controller.selected_algo
        if algo == "KNN":
            model = KNeighborsClassifier()
        elif algo == "SVM":
            model = SVC()
        elif algo == "Decision Tree":
            model = DecisionTreeClassifier()
        elif algo == "Linear Regression":
            model = LinearRegression()
        else:
            raise ValueError("Invalid algorithm")

        model.fit(X_train, y_train)
        controller.model = model
        y_pred = model.predict(X_test)
        controller.y_pred = y_pred

        if task == "classification":
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            return f"Accuracy: {acc:.2f}\nPrecision: {prec:.2f}\nRecall: {rec:.2f}\nF1 Score: {f1:.2f}\nConfusion Matrix:\n{cm}", cm
        else:
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            return f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nRÂ²: {r2:.2f}", None

def predict(controller, input_data):
    input_array = np.array([input_data])
    input_array = controller.scaler.transform(input_array)
    return controller.model.predict(input_array)[0]

# GUI Classes
class MLAnalyzerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ML Analyzer: Predict & Classify Any Dataset")
        self.geometry("900x600")
        self.configure(bg="#f0f0f0")
        self.resizable(True, True)

        # Container for pages
        container = tk.Frame(self, bg="#f0f0f0")
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = tk.Label(self, textvariable=self.status_var, bg="#f0f0f0", font=("Helvetica", 10), anchor="w")
        self.status_bar.pack(side="bottom", fill="x", padx=10, pady=5)

        # Initialize pages
        self.frames = {}
        for F in (WelcomePage, UploadPage, TrainPage):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        # Shared data
        self.df = None
        self.selected_task = None
        self.selected_algo = None
        self.target_column = None
        self.model = None
        self.label_encoders = {}
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

        self.show_frame("WelcomePage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()
        self.status_var.set(f"Current Page: {page_name}")

    def set_status(self, message, color="black"):
        self.status_var.set(message)
        self.status_bar.configure(fg=color)

class WelcomePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#f0f0f0")
        self.controller = controller

        # Style configuration
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 12), padding=10)
        style.configure("Green.TButton", background="#21A456", foreground="white")
        style.configure("Title.TLabel", font=("Helvetica", 20, "bold"))
        style.configure("Subtitle.TLabel", font=("Helvetica", 14))

        # Center frame
        center_frame = tk.Frame(self, bg="#f0f0f0")
        center_frame.place(relx=0.5, rely=0.5, anchor="center")

        # Title and subtitle
        ttk.Label(center_frame, text="ML Analyzer", style="Title.TLabel").pack(pady=20)
        ttk.Label(center_frame, text="Predict & Classify Any Dataset ", style="Subtitle.TLabel").pack(pady=5)
        ttk.Button(
            center_frame, text="Start", style="TButton",
            command=lambda: controller.show_frame("UploadPage")
        ).pack(pady=20)

class UploadPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#f0f0f0")
        self.controller = controller
        self.create_widgets()

    def create_widgets(self):
        # Style
        style = ttk.Style()
        style.configure("TLabel", font=("Helvetica", 10))
        style.configure("TButton", font=("Helvetica", 10), padding=5)
        style.configure("Treeview", font=("Helvetica", 10))
        style.configure("Treeview.Heading", font=("Helvetica", 10, "bold"))

        # Main container
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Title
        ttk.Label(main_frame, text="Configure Dataset & Model", font=("Helvetica", 16, "bold")).pack(pady=10)

        # Upload section
        upload_frame = ttk.LabelFrame(main_frame, text="Upload Dataset", padding=5)
        upload_frame.pack(fill="x", padx=5, pady=5)
        ttk.Button(upload_frame, text="Upload CSV", command=self.load_csv).pack(pady=5)

        # Columns and target selection
        config_frame = ttk.Frame(main_frame)
        config_frame.pack(fill="x", padx=5, pady=5)

        # Columns display
        columns_frame = ttk.LabelFrame(config_frame, text="Columns", padding=5)
        columns_frame.pack(side="left", fill="y", padx=5)
        self.columns_listbox = tk.Listbox(columns_frame, height=6, width=30, font=("Helvetica", 10))
        self.columns_listbox.pack(side="left", fill="y")
        scrollbar = ttk.Scrollbar(columns_frame, orient="vertical", command=self.columns_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.columns_listbox.config(yscrollcommand=scrollbar.set)

        # Configuration options
        options_frame = ttk.Frame(config_frame)
        options_frame.pack(side="left", fill="x", expand=True, padx=5)

        # Target column
        target_frame = ttk.LabelFrame(options_frame, text="Target Column", padding=5)
        target_frame.pack(fill="x", pady=2)
        self.target_var = tk.StringVar()
        self.target_dropdown = ttk.Combobox(target_frame, textvariable=self.target_var, state="readonly", width=40)
        self.target_dropdown.pack(pady=2)

        # Task selection
        task_frame = ttk.LabelFrame(options_frame, text="Task Type", padding=5)
        task_frame.pack(fill="x", pady=2)
        self.task_var = tk.StringVar()
        ttk.Radiobutton(task_frame, text="Classification", variable=self.task_var, value="classification", command=self.update_algo_options).pack(side="left", padx=10)
        ttk.Radiobutton(task_frame, text="Regression", variable=self.task_var, value="regression", command=self.update_algo_options).pack(side="left", padx=10)
        ttk.Radiobutton(task_frame, text="Clustering", variable=self.task_var, value="clustering", command=self.update_algo_options).pack(side="left", padx=10)

        # Algorithm selection
        algo_frame = ttk.LabelFrame(options_frame, text="Algorithm", padding=5)
        algo_frame.pack(fill="x", pady=2)
        self.algo_var = tk.StringVar()
        self.algo_dropdown = ttk.Combobox(algo_frame, textvariable=self.algo_var, state="readonly", width=40)
        self.algo_dropdown.pack(pady=2)

        # Data preview
        preview_frame = ttk.LabelFrame(main_frame, text="Data Preview", padding=5)
        preview_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.tree = ttk.Treeview(preview_frame, show="headings", height=5)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(preview_frame, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.tree.config(yscrollcommand=scrollbar.set)

        # Navigation buttons
        button_frame = ttk.Frame(self, padding=10)
        button_frame.pack(side="bottom", fill="x")
        ttk.Button(button_frame, text="Reset", command=self.reset).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Back", command=lambda: self.controller.show_frame("WelcomePage")).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Continue", command=self.go_to_train_page).pack(side="right", padx=5)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.controller.df = pd.read_csv(file_path)
                self.columns_listbox.delete(0, tk.END)
                columns = list(self.controller.df.columns)
                for col in columns:
                    self.columns_listbox.insert(tk.END, col)
                self.target_dropdown['values'] = columns
                self.target_dropdown.set(columns[-1])
                self.display_data_preview()
                preprocess_data(self.controller)
                self.controller.set_status("Dataset loaded successfully", "green")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")
                self.controller.set_status("Error loading dataset", "red")

    def display_data_preview(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        if self.controller.df is not None:
            df = self.controller.df.head()
            self.tree["columns"] = list(df.columns)
            for col in df.columns:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=100, anchor="center")
            for _, row in df.iterrows():
                self.tree.insert("", tk.END, values=list(row))

    def update_algo_options(self):
        task = self.task_var.get()
        self.controller.selected_task = task
        if task == "classification":
            self.algo_dropdown['values'] = ["KNN", "SVM", "Decision Tree"]
        elif task == "regression":
            self.algo_dropdown['values'] = ["Linear Regression"]
        elif task == "clustering":
            self.algo_dropdown['values'] = ["K-Means"]
        self.algo_dropdown.set("")

    def reset(self):
        self.controller.df = None
        self.controller.selected_task = None
        self.controller.selected_algo = None
        self.controller.target_column = None
        self.columns_listbox.delete(0, tk.END)
        self.target_dropdown.set("")
        self.target_dropdown['values'] = []
        self.task_var.set("")
        self.algo_dropdown.set("")
        self.algo_dropdown['values'] = []
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.controller.set_status("Configuration reset", "blue")

    def go_to_train_page(self):
        self.controller.selected_algo = self.algo_var.get()
        self.controller.target_column = self.target_var.get()
        
        # For clustering, target column is optional
        if self.controller.selected_task == "clustering":
            if self.controller.df is not None and self.controller.selected_algo:
                self.controller.frames["TrainPage"].prepare_inputs()
                self.controller.show_frame("TrainPage")
                self.controller.set_status("Moved to training page", "green")
                return
        else:
            # For classification/regression, we need all fields
            if self.controller.df is not None and self.controller.selected_task and self.controller.selected_algo and self.controller.target_column:
                self.controller.frames["TrainPage"].prepare_inputs()
                self.controller.show_frame("TrainPage")
                self.controller.set_status("Moved to training page", "green")
                return
        
        messagebox.showwarning("Missing Information", "Please upload a dataset and select all required options.")
        self.controller.set_status("Missing configuration", "red")

class TrainPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#f0f0f0")
        self.controller = controller
        self.entries = []
        self.model_trained = False
        self.canvas = None
        self.create_widgets()

    def create_widgets(self):
        # Style
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 10), padding=5)
        style.configure("TLabel", font=("Helvetica", 10))

        # Title
        ttk.Label(self, text="Train Model & Predict", font=("Helvetica", 16, "bold")).pack(pady=10)

        # Train button
        self.train_button = ttk.Button(self, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=5)

        # Evaluation output
        eval_frame = ttk.LabelFrame(self, text="Evaluation Results", padding=10)
        eval_frame.pack(fill="both", padx=15, pady=5)
        self.result_text = tk.Text(eval_frame, height=6, width=80, font=("Helvetica", 10), wrap="word")
        self.result_text.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(eval_frame, orient="vertical", command=self.result_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.result_text.config(state="disabled", yscrollcommand=scrollbar.set)

        # Prediction input frame
        self.prediction_frame = ttk.LabelFrame(self, text="Prediction Inputs", padding=5)
        # Not packed until training

        # Plot frame
        self.plot_frame = ttk.LabelFrame(self, text="Visualization", padding=5)
        # Not packed until needed

        # Buttons frame
        button_frame = ttk.Frame(self, padding=10)
        button_frame.pack(side="bottom", fill="x")
        ttk.Button(button_frame, text="Clear Results", command=self.clear_results).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Back", command=self.on_back).pack(side="left", padx=5)
        self.predict_button = ttk.Button(button_frame, text="Predict", command=self.predict)
        self.predict_button.pack(side="right", padx=5)

    def prepare_inputs(self):
        for widget in self.prediction_frame.winfo_children():
            widget.destroy()
        
        df = self.controller.df
        if self.controller.selected_task == "clustering":
            # For clustering, use all columns as features
            feature_cols = list(df.columns)
        else:
            # For classification/regression, exclude target column
            target_col = self.controller.target_column
            feature_cols = [col for col in df.columns if col != target_col]
            
        self.entries = []
        for i in range(0, len(feature_cols), 2):
            row_frame = ttk.Frame(self.prediction_frame)
            row_frame.pack(fill="x", padx=5, pady=2)
            # First input in the row
            col = feature_cols[i]
            frame1 = ttk.Frame(row_frame)
            frame1.pack(side="left", fill="x", expand=True, padx=5)
            ttk.Label(frame1, text=col, width=20, anchor="w").pack(side="left")
            ent1 = ttk.Entry(frame1, width=30)
            ent1.pack(side="left")
            self.entries.append((col, ent1))
            # Second input in the row, if it exists
            if i + 1 < len(feature_cols):
                col = feature_cols[i + 1]
                frame2 = ttk.Frame(row_frame)
                frame2.pack(side="left", fill="x", expand=True, padx=5)
                ttk.Label(frame2, text=col, width=20, anchor="w").pack(side="left")
                ent2 = ttk.Entry(frame2, width=30)
                ent2.pack(side="left")
                self.entries.append((col, ent2))

    def show_prediction_section(self):
        if not self.model_trained:
            self.prepare_inputs()
            self.prediction_frame.pack(fill="x", padx=15, pady=5)
            self.model_trained = True

    def clear_canvas(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None

    def show_plot(self, fig):
        self.clear_canvas()
        self.plot_frame.pack(fill="both", padx=15, pady=5)
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def train_model(self):
        self.train_button.config(state="disabled")
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", "Training model, please wait...")
        self.result_text.config(state="disabled")
        self.controller.set_status("Training model...", "blue")
        self.update()
        
        try:
            result, additional_data = train_model(self.controller)
            self.result_text.config(state="normal")
            self.result_text.delete("1.0", "end")
            self.result_text.insert("end", result)
            self.result_text.config(state="disabled")
            self.show_prediction_section()
            
            if self.controller.selected_task == "classification":
                # Generate Confusion Matrix heatmap
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(additional_data, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title("Confusion Matrix")
                self.show_plot(fig)
                plt.close(fig)
            elif self.controller.selected_task == "regression":
                # Generate Scatter Plot: Predicted vs Actual
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(self.controller.y_test, self.controller.y_pred, color='blue', alpha=0.5)
                ax.plot([self.controller.y_test.min(), self.controller.y_test.max()], 
                        [self.controller.y_test.min(), self.controller.y_test.max()], 'r--', lw=2)
                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Predicted Values')
                ax.set_title('Predicted vs Actual')
                self.show_plot(fig)
                plt.close(fig)
            elif self.controller.selected_task == "clustering":
                # Generate Scatter Plot for clusters
                df = self.controller.df
                if len(df.columns) >= 2:
                    # Use first two columns for scatter plot
                    x_col = df.columns[0]
                    y_col = df.columns[1]
                    
                    fig, ax = plt.subplots(figsize=(18, 8))
                    scatter = ax.scatter(df[x_col], df[y_col], c=additional_data, cmap='viridis',s=100)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title('K-Means Clustering (k=3)') 
                    plt.colorbar(scatter, ax=ax, label='Cluster')
                    self.show_plot(fig)
                    plt.savefig("clustering_plot.png")  # Save the plot
                    self.show_plot(fig)
                    plt.close(fig)
                else:
                    messagebox.showinfo("Info", "Dataset has less than 2 features - cannot plot clusters")
            
            self.controller.set_status("Training completed", "green")
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            self.result_text.config(state="normal")
            self.result_text.delete("1.0", "end")
            self.result_text.insert("end", f"Error: {str(e)}")
            self.result_text.config(state="disabled")
            self.controller.set_status("Training failed", "red")
        finally:
            self.train_button.config(state="normal")

    def predict(self):
        try:
            input_data = []
            for col, ent in self.entries:
                value = ent.get().strip()
                if not value:
                    raise ValueError(f"Input for {col} is empty")
                if col in self.controller.label_encoders:
                    le = self.controller.label_encoders[col]
                    if value in le.classes_:
                        value = le.transform([value])[0]
                    else:
                        raise ValueError(f"Value '{value}' not in known categories for column '{col}'")
                else:
                    value = float(value)
                input_data.append(value)
                
            prediction = predict(self.controller, input_data)
            
            if self.controller.selected_task == "classification" and self.controller.target_column in self.controller.label_encoders:
                le = self.controller.label_encoders[self.controller.target_column]
                prediction = le.inverse_transform([int(prediction)])[0]
                message = f"Predicted class: {prediction}"
            elif self.controller.selected_task == "clustering":
                message = f"Predicted cluster: {prediction}"
            else:
                message = f"Predicted value: {prediction}"
                
            messagebox.showinfo("Prediction", message)
            self.controller.set_status("Prediction completed", "green")
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            self.controller.set_status("Prediction failed", "red")

    def clear_results(self):
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.config(state="disabled")
        self.clear_canvas()
        self.plot_frame.pack_forget()
        for widget in self.prediction_frame.winfo_children():
            widget.destroy()
        self.prediction_frame.pack_forget()
        self.model_trained = False
        self.controller.set_status("Results cleared", "blue")

    def on_back(self):
        self.clear_results()
        self.controller.show_frame("UploadPage")

# Main Execution
if __name__ == "__main__":
    app = MLAnalyzerApp()
    app.mainloop()

