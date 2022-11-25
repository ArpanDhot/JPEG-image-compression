from tkinter import *
from tkinter import messagebox, filedialog
import threading
from en_de_code import encodeImage,decodeImg

class Main:
    # Creating the window
    master = Tk()

    # Colors in window
    backgroundColor = "#D3D0CF"
    buttonBackgroundColor = "grey"
    foregroundColor = "#A49B96"
    textColor = "#ffffff"
    textColor2 = "red"
    textColor3 = "blue"

    # Getting the windows sizes
    winHeight = master.winfo_screenheight()
    winWidth = master.winfo_screenwidth()

    # variables
    report = StringVar()
    imageTo = StringVar()
    imagePath = StringVar()
    encodeOption = IntVar()
    compression = IntVar()
    encodeOption.set(1)
    compression.set(90)

    # Creating labels font
    headerFont = ("Calibre", int(winHeight / 15), "bold")
    buttonFont = ("Calibre", int(winHeight / 50))
    simpleTextFont = ("Calibre", int(winHeight / 53), "bold")
    simpleTextFont2 = ("Calibre", int(winHeight / 60))
    textEntryFont = ("Calibre", int(winHeight / 60))

    # Labels
    labelOne = Label(master, bg=backgroundColor, fg=foregroundColor, text="FILE COMPRESSOR", font=headerFont)
    labelTwo = Label(master, bg=foregroundColor, fg=textColor, text="")
    labelFour = Label(master, textvariable=report, bg=foregroundColor, fg=textColor, font=simpleTextFont2, anchor=NW,
                      justify=LEFT)
    labelTen = Label(master, bg=foregroundColor, fg=textColor, text="Compression Strength", font=simpleTextFont2, anchor=W)
    labelEleven = Label(master, bg=foregroundColor, fg=textColor2, text="------ Compression strength " + str("-" * 50),
                     font=simpleTextFont2)

    textEntryOne = Text(master, font=textEntryFont, bg="white", fg="black", bd=0)
    textEntryTwo = Text(master, font=textEntryFont, bg="white", fg="black", bd=0)

    buttonOne = Button(master, bg=buttonBackgroundColor, fg=textColor, text="Image", borderwidth=0, font=buttonFont)
    buttonTwo = Button(master, bg=buttonBackgroundColor, fg=textColor, text="Save at", borderwidth=0, font=buttonFont)
    buttonThree = Button(master, bg=buttonBackgroundColor, fg=textColor, text="COMPRESS", borderwidth=0, font=buttonFont)

    drop = Text(master, font=textEntryFont, bg="white", fg="black", bd=0)
    drop.insert(INSERT,str(compression.get()))

    def __init__(self):
        # Getting the size of the screen and removing 100 pixels from it to make a bit smaller
        self.master.geometry("{1}x{0}+40+20".format(self.winHeight - 100, self.winWidth - 100))
        self.master.resizable(0, 0)
        self.master.configure(bg=self.backgroundColor)

        Main.report.set("Enter the information and press compress ... ")

        # destroy the window if "Escape key is pressed"
        self.master.bind("<Escape>", self.killWindow)
        self.placePage()
        self.master.mainloop()

    def getFilePath(self, x):
        if x == 1:
            Main.imagePath.set(filedialog.askopenfilename())
            self.textEntryOne.delete("1.0", "end")
            self.textEntryOne.insert(INSERT, Main.imagePath.get())
        if x == 2:
            Main.imageTo.set(filedialog.askdirectory())
            self.textEntryTwo.delete("1.0", "end")
            self.textEntryTwo.insert(INSERT, Main.imageTo.get())

    def ImageCompressionEngine(self):
        Main.report.set("")
        if Main.imagePath.get() != "" and Main.imageTo.get() != "":
            Main.compression.set(int(self.drop.get("1.0", "end-1c")))

            if bool(Main.encodeOption.get()):
                Main.report.set(Main.report.get()+"\n[+] Compression started..")
                Main.report.set(Main.report.get()+"\n[+] Huffman Encoding started..")
                encode = encodeImage(Main.imagePath.get(),Main.imageTo.get(),Main.report)
                decodeImg(encode,Main.imageTo.get(),Main.report,Main.compression.get()/100)
        else:
            messagebox.showinfo("showinfo", "Please select Both paths")

    # Methods to perform task when window is destroyed
    def killWindow(self, event):
        self.master.destroy()  # destroying the window

    # Creating the login page
    def placePage(self):
        self.buttonTwo.config(command=lambda: [self.getFilePath(2)])
        self.buttonOne.config(command=lambda: [self.getFilePath(1)])
        self.buttonThree.config(command=lambda: [threading.Thread(target=self.ImageCompressionEngine).start()])

        self.labelOne.place(relx=0.2, rely=0, relwidth=0.6, relheight=0.16)
        self.labelTwo.place(relx=0.45 + 0.05, rely=0.2, relwidth=0.45, relheight=0.76)
        self.labelFour.place(relx=0.04, rely=0.2, relwidth=0.45, relheight=0.76)

        self.textEntryOne.place(relx=0.48 + 0.05, rely=0.24, relwidth=0.3, relheight=0.04)
        self.textEntryTwo.place(relx=0.48 + 0.05, rely=0.32, relwidth=0.3, relheight=0.04)

        self.labelEleven.place(relx=0.48 + 0.05, rely=0.4, relwidth=0.3, relheight=0.04)
        self.labelTen.place(relx=0.48 + 0.05, rely=0.48, relwidth=0.145, relheight=0.04)
        self.drop.place(relx=0.48 + 0.2, rely=0.48, relwidth=0.145, relheight=0.04)

        self.buttonOne.place(relx=0.84, rely=0.24, relwidth=0.08, relheight=0.04)
        self.buttonTwo.place(relx=0.84, rely=0.32, relwidth=0.08, relheight=0.04)
        self.buttonThree.place(relx=0.84, rely=0.88, relwidth=0.1, relheight=0.06)



# Calling the Main class that contain the GUI and DBS commands
if __name__ == '__main__':
    m = Main()























