import pandas as pd
import os
import wx
import wx.adv
import threading
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas


class LoadingDialog(wx.Dialog):
    def __init__(self, parent, message="Processing..."):
        super().__init__(parent, title="Please wait", size=(250, 100))
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        text = wx.StaticText(panel, label=message, style=wx.ALIGN_CENTER)
        vbox.Add(text, proportion=1, flag=wx.ALL | wx.ALIGN_CENTER, border=20)

        panel.SetSizer(vbox)
        self.CenterOnParent()

def read_file_to_dataframe(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in [".xls", ".xlsx"]:
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

class FileSelector(wx.Frame):
    def __init__(self, title="Select a CSV or Excel file"):
        super().__init__(None, title=title, size=(500, 100))
        panel = wx.Panel(self)
        hbox = wx.BoxSizer(wx.HORIZONTAL)

        file_types = "CSV and Excel files (*.csv;*.xls;*.xlsx)|*.csv;*.xls;*.xlsx"
        self.file_picker = wx.FilePickerCtrl(panel, message=title, wildcard=file_types)
        hbox.Add(self.file_picker, proportion=1, flag=wx.EXPAND | wx.ALL, border=10)

        panel.SetSizer(hbox)

        self.selected_path = None
        self.file_picker.Bind(wx.EVT_FILEPICKER_CHANGED, self.on_file_selected)

    def on_file_selected(self, event):
        path = self.file_picker.GetPath()
        if path:
            self.selected_path = path
            self.Close()

def select_file_via_gui(window_title="Select a CSV or Excel file"):
    app = wx.App()
    frame = FileSelector(title=window_title)
    frame.Show()
    app.MainLoop()

    if frame.selected_path:
        try:
            df = read_file_to_dataframe(frame.selected_path)
            return df
        except Exception as e:
            wx.MessageBox(f"Failed to read file:\n{e}", "Error", wx.OK | wx.ICON_ERROR)
            return None
    else:
        return None

class ColumnSelector(wx.Frame):
    def __init__(self, df, title="Select a Column"):
        super().__init__(None, title=title, size=(400, 300))
        self.df = df
        self.selected_column = None

        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        self.listbox = wx.ListBox(panel, choices=list(self.df.columns))
        vbox.Add(self.listbox, proportion=1, flag=wx.EXPAND | wx.ALL, border=10)

        self.btn_select = wx.Button(panel, label="Select Column")
        vbox.Add(self.btn_select, flag=wx.ALIGN_CENTER | wx.ALL, border=10)

        panel.SetSizer(vbox)

        self.btn_select.Bind(wx.EVT_BUTTON, self.on_select)

    def on_select(self, event):
        selection = self.listbox.GetSelection()
        if selection == wx.NOT_FOUND:
            wx.MessageBox("Please select a column.", "Warning", wx.OK | wx.ICON_WARNING)
        else:
            self.selected_column = self.listbox.GetString(selection)
            wx.MessageBox(f"You selected: {self.selected_column}", "Selection", wx.OK | wx.ICON_INFORMATION)
            self.Close()

def select_column_from_dataframe(df, window_title="Select a Column"):
    app = wx.App()
    frame = ColumnSelector(df, title=window_title)
    frame.Show()
    app.MainLoop()
    return frame.selected_column


class MultiSelectCheckList(wx.Frame):
    def __init__(self, items, preselected_items=None, title="Select Items"):
        super().__init__(None, title=title, size=(400, 300))

        self.selected_items = []

        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        self.checklist = wx.CheckListBox(panel, choices=items)
        vbox.Add(self.checklist, proportion=1, flag=wx.EXPAND | wx.ALL, border=10)

        # Pre-check items that appear in preselected_items
        if preselected_items:
            items_lower = [item.lower() for item in items]
            for pre_item in preselected_items:
                # Case-insensitive match, adjust as needed
                try:
                    idx = items_lower.index(pre_item.lower())
                    self.checklist.Check(idx)
                except ValueError:
                    pass  # item not in the list, ignore

        btn_select = wx.Button(panel, label="Select")
        vbox.Add(btn_select, flag=wx.ALIGN_CENTER | wx.ALL, border=10)

        panel.SetSizer(vbox)

        btn_select.Bind(wx.EVT_BUTTON, self.on_select)

    def on_select(self, event):
        checked_indices = [i for i in range(self.checklist.GetCount()) if self.checklist.IsChecked(i)]
        if not checked_indices:
            wx.MessageBox("Please select at least one item.", "Warning", wx.OK | wx.ICON_WARNING)
            return
        self.selected_items = [self.checklist.GetString(i) for i in checked_indices]
        self.Close()

def select_multiple_items(items, preselected_items=None, window_title="Select Items"):
    app = wx.App()
    frame = MultiSelectCheckList(items, preselected_items=preselected_items, title=window_title)
    frame.Show()
    app.MainLoop()
    return frame.selected_items


#Correr una funcion enviada al mandar llamar el objeto
class RerunPrompt(wx.Frame):
    def __init__(self, function_to_call, title="", message="Do you want to rerun the process?"):
        super().__init__(None, title=title, size=(600, 300))
        self.function_to_call = function_to_call

        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        label = wx.StaticText(panel, label=message)
        vbox.Add(label, flag=wx.ALIGN_CENTER | wx.ALL, border=15)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        btn_yes = wx.Button(panel, label="Yes")
        btn_no = wx.Button(panel, label="No")
        hbox.Add(btn_yes, flag=wx.RIGHT, border=10)
        hbox.Add(btn_no)

        vbox.Add(hbox, flag=wx.ALIGN_CENTER | wx.BOTTOM, border=10)

        panel.SetSizer(vbox)

        btn_yes.Bind(wx.EVT_BUTTON, self.on_yes)
        btn_no.Bind(wx.EVT_BUTTON, self.on_no)

    def on_yes(self, event):
        self.Hide()  # Hide the main frame
        self.run_with_loading()
        return True

    def on_no(self, event):
        self.Close()
        return False

    def run_with_loading(self):
        loading_dialog = LoadingDialog(self, message="Running process...")
        loading_dialog.Show()

        def task():
            try:
                self.function_to_call()
            finally:
                wx.CallAfter(loading_dialog.Destroy)
                wx.CallAfter(self.Close)

        threading.Thread(target=task, daemon=True).start()

def ask_to_rerun(function_to_call, window_title="", prompt_message="Do you want to rerun the process?"):
    app = wx.App()
    frame = RerunPrompt(function_to_call=function_to_call, title=window_title, message=prompt_message)
    frame.Show()
    app.MainLoop()


#Regresa un booleano dependiendo de la eleccion
class RerunOptionPrompt(wx.Frame):
    def __init__(self, title="Rerun Process?", message="Do you want to rerun the process?"):
        super().__init__(None, title=title, size=(300, 150))
        self.result = None  # Will hold True (Yes) or False (No)

        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        label = wx.StaticText(panel, label=message)
        vbox.Add(label, flag=wx.ALIGN_CENTER | wx.ALL, border=15)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        btn_yes = wx.Button(panel, label="Yes")
        btn_no = wx.Button(panel, label="No")
        hbox.Add(btn_yes, flag=wx.RIGHT, border=10)
        hbox.Add(btn_no)

        vbox.Add(hbox, flag=wx.ALIGN_CENTER | wx.BOTTOM, border=10)
        panel.SetSizer(vbox)

        btn_yes.Bind(wx.EVT_BUTTON, self.on_yes)
        btn_no.Bind(wx.EVT_BUTTON, self.on_no)

        self.Bind(wx.EVT_CLOSE, self.on_close)

    def on_yes(self, event):
        self.result = True
        self.Close()

    def on_no(self, event):
        self.result = False
        self.Close()

    def on_close(self, event):
        # Default to False if window is closed without button click
        if self.result is None:
            self.result = False
        self.Destroy()

def ask_to_rerun_opt(window_title="Rerun?", prompt_message="Do you want to rerun the process?") -> bool:
    app = wx.App(False)
    frame = RerunOptionPrompt(title=window_title, message=prompt_message)
    frame.Show()
    app.MainLoop()
    return frame.result



class ValueInputDialog(wx.Frame):
    def __init__(self, prompt="Enter a value:", title="Input Required"):
        super().__init__(None, title=title, size=(350, 150))

        self.value = None

        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        self.label = wx.StaticText(panel, label=prompt)
        vbox.Add(self.label, flag=wx.ALL | wx.ALIGN_CENTER, border=10)

        self.text_ctrl = wx.TextCtrl(panel)
        vbox.Add(self.text_ctrl, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=10)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        btn_ok = wx.Button(panel, label="OK")
        btn_cancel = wx.Button(panel, label="Cancel")
        hbox.Add(btn_ok, flag=wx.RIGHT, border=10)
        hbox.Add(btn_cancel)

        vbox.Add(hbox, flag=wx.ALIGN_CENTER | wx.ALL, border=10)

        panel.SetSizer(vbox)

        btn_ok.Bind(wx.EVT_BUTTON, self.on_ok)
        btn_cancel.Bind(wx.EVT_BUTTON, self.on_cancel)

    def on_ok(self, event):
        self.value = self.text_ctrl.GetValue()
        self.Close()

    def on_cancel(self, event):
        self.Close()

def get_user_input(prompt="Enter a value:", window_title="Input Required"):
    app = wx.App()
    frame = ValueInputDialog(prompt=prompt, title=window_title)
    frame.Show()
    app.MainLoop()
    return frame.value


class PlotViewer(wx.Frame):
    def __init__(self, fig, title="Plot Viewer"):
        super().__init__(None, title=title, size=(640, 500))

        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Embed the matplotlib figure
        self.canvas = FigureCanvas(panel, -1, fig)
        vbox.Add(self.canvas, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)

        # Add the OK button
        btn_ok = wx.Button(panel, label="OK")
        vbox.Add(btn_ok, flag=wx.ALIGN_CENTER | wx.ALL, border=10)

        btn_ok.Bind(wx.EVT_BUTTON, self.on_close)

        panel.SetSizer(vbox)
        self.Center()

    def on_close(self, event):
        self.Close()

def show_plot_in_wx_gui(fig, window_title="Plot Viewer"):
    app = wx.App()
    frame = PlotViewer(fig, title=window_title)
    frame.Show()
    app.MainLoop()




class EventPrompt(wx.Frame):
    def __init__(self, title="Input with Dates"):
        super().__init__(parent=None, title=title, size=(400, 300))

        self.text = None
        self.startdate = None
        self.enddate = None


        panel = wx.Panel(self)

        # Layout using a vertical BoxSizer
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Text input
        self.text_label = wx.StaticText(panel, label="Nombre:")
        self.text_ctrl = wx.TextCtrl(panel)

        # Start date
        self.start_date_label = wx.StaticText(panel, label="Fecha de Inicio:")
        self.start_date_picker = wx.adv.DatePickerCtrl(panel)

        # End date
        self.end_date_label = wx.StaticText(panel, label="Fecha de Fin:")
        self.end_date_picker = wx.adv.DatePickerCtrl(panel)

        # Add widgets to sizer
        vbox.Add(self.text_label, flag=wx.LEFT | wx.TOP, border=10)
        vbox.Add(self.text_ctrl, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=10)
        vbox.AddSpacer(10)

        vbox.Add(self.start_date_label, flag=wx.LEFT, border=10)
        vbox.Add(self.start_date_picker, flag=wx.LEFT | wx.RIGHT, border=10)
        vbox.AddSpacer(10)

        vbox.Add(self.end_date_label, flag=wx.LEFT, border=10)
        vbox.Add(self.end_date_picker, flag=wx.LEFT | wx.RIGHT, border=10)
        vbox.AddSpacer(20)
        
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        btn_ok = wx.Button(panel, label="OK")
        btn_cancel = wx.Button(panel, label="Cancel")
        hbox.Add(btn_ok, flag=wx.RIGHT, border=10)
        hbox.Add(btn_cancel)

        vbox.Add(hbox, flag=wx.ALIGN_CENTER | wx.ALL, border=10)

        panel.SetSizer(vbox)

        btn_ok.Bind(wx.EVT_BUTTON, self.on_ok)
        btn_cancel.Bind(wx.EVT_BUTTON, self.on_cancel)

    def on_ok(self, event):
        self.text = self.text_ctrl.GetValue()
        self.startdate = self.start_date_picker.GetValue().FormatISODate()
        self.enddate = self.end_date_picker.GetValue().FormatISODate()
        self.Close()

    def on_cancel(self, event):
        self.Close()

def event_info(window_title="Add Event"):
    app = wx.App()
    frame = EventPrompt(title=window_title)
    frame.Show()
    app.MainLoop()
    return frame.text, frame.startdate, frame.enddate