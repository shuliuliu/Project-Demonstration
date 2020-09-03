# Python | Loan calculator using Tkinter
# Python offers multiple options for developing GUI (Graphical User Interface). Out of all the GUI methods, tkinter is most commonly used method. It is a standard Python interface to the Tk GUI toolkit shipped with Python. Python with tkinter outputs the fastest and easiest way to create the GUI applications. Creating a GUI using tkinter is an easy task.
from tkinter import *
from tkinter.ttk import *

class MortgageCalculator:

    def __init__(self):
        # create a window
        window = Tk()
        # set title
        window.title('Mortgage Calculator')

        # create the input boxes for "Compute Payment" and take inputs
        Label(window, text='Sales Price').\
            grid(row=1,column=1,sticky=W) # create input box
        self.salesPriceVar = StringVar() # taking the input
        Entry(window, textvariable=self.salesPriceVar, justify=RIGHT, width=10).\
            grid(row=1, column=2)

        Label(window, text='Down Payment Percentage').\
            grid(row=2,column=1,sticky=W)
        self.downPaymentPctVar = StringVar()
        Entry(window, textvariable=self.downPaymentPctVar, justify=RIGHT, width=10).\
            grid(row=2, column=2)

        Label(window, text='Annual Interest Rate').\
            grid(row=3,column=1,sticky=W)
        self.annualInterestRateVar = StringVar()
        Entry(window, textvariable=self.annualInterestRateVar, justify=RIGHT, width=10).\
            grid(row=3, column=2)

        Label(window, text='Number of Years').\
            grid(row=4,column=1,sticky=W)
        self.numberOfYearsVar = StringVar()
        Entry(window, textvariable=self.numberOfYearsVar, justify=RIGHT, width=10).\
            grid(row=4, column=2)

        # create the "Compute Payment" button
        btComputePayment = Button(window, text='Compute Payment', command=self.computePayment).\
            grid(row=5, column=5, sticky=E)

        # create the input boxes for "Compute Loan" and take inputs
        Label(window, text='Affordable Payment/mo.').\
            grid(row=7, column=1, sticky=W)
        self.monthlyPaymentVar2 = StringVar()
        Entry(window, textvariable=self.monthlyPaymentVar2, justify=RIGHT, width=10).\
            grid(row=7, column=2)

        # create the "Compute Loan" button
        btComputeLoan = Button(window, text='Compute Loan', command=self.computeLoan). \
            grid(row=8, column=5, sticky=E)

        # create the "Compute Payment" output boxes and generate the outputs
        Label(window, text='Loan Amount').\
            grid(row=1, column=4, sticky=W)
        self.loanAmountVar = StringVar()  # generating output
        lblLoanAmount = Label(window, textvariable=self.loanAmountVar).\
            grid(row=1, column=5, sticky=E)

        Label(window, text='Monthly Payment').\
            grid(row=2, column=4, sticky=W)
        self.monthlyPaymentVar = StringVar()
        lblMonthlyPayment = Label(window, textvariable=self.monthlyPaymentVar).\
            grid(row=2, column=5, sticky=E)

        Label(window, text='Total Payment').\
            grid(row=3, column=4, sticky=W)
        self.totalPaymentVar = StringVar()
        lblTotalPayment = Label(window, textvariable=self.totalPaymentVar).\
            grid(row=3, column=5, sticky=E)

        # create the "Compute Loan" output boxes and generate the output "Loanable Amount"
        Label(window, text='Loanable Amount').\
            grid(row=7, column=4, sticky=W)
        self.loanAmountVar2 = StringVar()
        lblLoanAmount2 = Label(window, textvariable=self.loanAmountVar2).\
            grid(row=7, column=5, sticky=E)

        # design the layout of the panel
        col_count, row_count = window.grid_size()
        print(row_count)
        for col in range(col_count):
            window.grid_columnconfigure(col, minsize=20)
        for row in range(row_count):
            window.grid_rowconfigure(row, minsize=20)
        # add two empty columns as spaces
        window.grid_columnconfigure(4, minsize=100)
        window.grid_columnconfigure(6, minsize=20)

        # create an event loop
        window.mainloop()


    def computePayment(self):
        loanAmount = float(self.salesPriceVar.get())*(1-float(self.downPaymentPctVar.get())/100)
        self.loanAmountVar.set(format(loanAmount,'10.2f'))

        monthlyPayment = self.getMonthlyPayment(
            float(self.salesPriceVar.get()),
            float(self.downPaymentPctVar.get())/100,
            float(self.annualInterestRateVar.get()) / (12*100),
            int(self.numberOfYearsVar.get())*12
        )
        self.monthlyPaymentVar.set(format(monthlyPayment,'10.2f'))

        totalPayment = float(self.monthlyPaymentVar.get())* 12 * int(self.numberOfYearsVar.get())
        self.totalPaymentVar.set(format(totalPayment,'10.2f'))


    def getMonthlyPayment(self,salesPrice,downPaymentPct,monthlyInterestRate,loanTerm):
        # calculate the monthly loan payment
        monthlyPayment = (salesPrice*(1-downPaymentPct)) * ((1+monthlyInterestRate) ** loanTerm) * (monthlyInterestRate) / ((1+monthlyInterestRate) ** loanTerm - 1)
        return monthlyPayment;
        root = Tk() # create the widget


    def computeLoan(self):
        loanAmount2 = self.getLoanAmount(
            float(self.monthlyPaymentVar2.get()),
            float(self.annualInterestRateVar.get()) / (12*100),
            int(self.numberOfYearsVar.get())*12
        )
        self.loanAmountVar2.set(format(loanAmount2, '10.2f'))

    def getLoanAmount(self, monthlyPayment, monthlyInterestRate, loanTerm):
        loanAmount = monthlyPayment * (1 - 1 / (1 + monthlyInterestRate) ** loanTerm) / monthlyInterestRate
        return loanAmount;
        root = Tk()  # create the widget


# call the class to run the calculator
MortgageCalculator()