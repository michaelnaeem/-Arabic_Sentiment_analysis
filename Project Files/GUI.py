from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, Label, filedialog, ttk
import Predict
import pandas as pd
bg_colour = '#3d6466'


root = Tk()
root.title("انطباع")
root.geometry("700x550")
root.resizable(0,0)


bg_image_att_1st = PhotoImage(file="assets/frame0/1st_Page.png")
analyse_att = PhotoImage(file="assets/frame0/Analyse.png")
upload_att = PhotoImage(file="assets/frame0/Upload_Button_Page.png")
exit_att = PhotoImage(file="assets/frame0/Exit_Button.png")
bg_image_att_2nd = PhotoImage(file="assets/frame1/2nd_Page.png")
back_att = PhotoImage(file="assets/frame1/back_button.png")
upload_icon_att = PhotoImage(file="assets/frame1/Upload_button.png")
nodata_att = PhotoImage(file="assets/frame1/no_data.png")
nodata1_att = PhotoImage(file="assets/frame1/no_data1.png")
def load_canvas1():
    prediction = "تحليلك هيكون هنا"
    pred_conf = "نسبة الثقة من النتيجة"
    canvas2.destroy()
    canvas1 = Canvas(
        root,
        bg = bg_colour,
        height = 550,
        width = 700,
        bd = 0,
        highlightthickness = 0,
        relief = "ridge"
        )
    canvas1.place(x = 0, y = 0)
    bg_image = canvas1.create_image(
        350,
        275,
        image=bg_image_att_1st
        )
    analyse_button = Button(
        canvas1,
        image = analyse_att,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: predict(input.get()),
        relief="flat"
        )
    analyse_button.place(
        x=450,
        y=336)

    upload_button = Button(
        canvas1,
        image = upload_att,
        borderwidth=0,
        highlightthickness=0,
        command=load_canvas2,
        relief="flat"
        )
    upload_button.place(
        x=288,
        y=336)

    exit_button = Button(
        canvas1,
        image = exit_att,
        borderwidth=0,
        highlightthickness=0,
        command=root.destroy,
        relief="flat"
        )
    exit_button.place(
        x=-6,
        y=460)

    input = Entry(
        canvas1,
        bd=0,
        highlightthickness=0
        )
    input.place(
        x=300.0,
        y=280.0,
        width=256.0,
        height=40.0
        )

    prediction_label = canvas1.create_text(
        275,
        430,
        anchor="nw",
        text=prediction,
        font=("IrishGrover Regular", 32 * -1)
        )

    prediction_confedence = canvas1.create_text(
        275,
        485,
        anchor="nw",
        text=pred_conf,
        font=("IrishGrover Regular", 27 * -1)
        )

    def update_progress(leng):
        for i in range(leng):
            progress_bar["value"] = i
            root.update_idletasks()
            root.after(20)

    def predict(sentence):
        global prediction, pred_conf, progress_bar
        prediction, pred_conf = Predict.predict(sentence)
        s = ttk.Style()
        s.theme_use('clam')
        if pred_conf < 50:
            pred_conf = 100 - pred_conf
            s.configure("red.Horizontal.TProgressbar", foreground="red", background="#90EE90")
        else: s.configure("red.Horizontal.TProgressbar", foreground="white", background="#FF0000")
        canvas1.itemconfig(tagOrId=prediction_label, text = prediction)
        canvas1.itemconfig(tagOrId=prediction_confedence, text = str(pred_conf) + "%")
        progress_bar = ttk.Progressbar(canvas1, orient="horizontal", mode="determinate", style="red.Horizontal.TProgressbar")
        progress_bar.place(
            x=238.0,
            y=468.0,
            width=240.0,
            height=10.0
            )
        update_progress(round(pred_conf))
def load_canvas2():
    canvas1.destroy()
    canvas2 = Canvas(
        root,
        bg = bg_colour,
        height = 550,
        width = 700,
        bd = 0,
        highlightthickness = 0,
        relief = "ridge"
        )
    canvas2.place(x = 0, y = 0)

    bg_image = canvas2.create_image(
        350,
        275,
        image=bg_image_att_2nd
        )

    back_button = Button(
        canvas2,
        image = back_att,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: load_canvas1(),
        relief="flat"
        )
    back_button.place(
        x=-6,
        y=434)

    upload_button = Button(
        canvas2,
        image = upload_icon_att,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: browse_files(),
        relief="flat"
        )
    upload_button.place(
        x=322,
        y=179)

    piechart = canvas2.create_image(
    565,
    385,
    image=nodata_att
    )

    wordcloud = canvas2.create_image(
    330,
    385,
    image=nodata1_att
    )

    def browse_files():
        global wordcloud_att, piechart_att
        filename = filedialog.askopenfilename(initialdir="/", title="Select a File", filetypes=(("Text files", "*.txt*"), ("All files", "*.*")))
        if filename:
            df = Predict.predict_file(filename)
            print("prediction done!")
            df[['text', 'class', 'confidence']].to_csv(filename[:-4]+"Cleaned.txt", encoding = 'utf-8', index = False, sep = '\t')
            Predict.piechart(df)
            print("piechart done!")
            Predict.PreProcessing.wordcloud(df)
            print("wordcloud done!")
            wordcloud_att = PhotoImage(file="assets/frame1/wordcloud.png")
            piechart_att = PhotoImage(file="assets/frame1/pie_chart.png")
            canvas2.itemconfig(tagOrId=piechart, image = piechart_att)
            canvas2.itemconfig(tagOrId=wordcloud, image = wordcloud_att)

canvas1 = Canvas(root)
canvas2 = Canvas(root)
load_canvas1()
root.mainloop()
