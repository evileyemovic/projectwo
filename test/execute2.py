from tkinter import *
import find
import argparse
import control
import quiz


def anal(ent1):
    public_cards = []
    a = ent1.get()
    tmp = 0
    b = ""
    for i in a:
        b = b + i
        tmp += 1
        if tmp % 2 == 0:
            public_cards.append(b)
            b = ""
    find.traincs(args, public_cards)

def anal2(ent1):
    public_cards = []
    a = ent1.get()
    tmp = 0
    b = ""
    for i in a:
        b = b + i
        tmp += 1
        if tmp % 2 == 0:
            public_cards.append(b)
            b = ""
    find.trainma(args, public_cards)



def strategy1():
    strg = Tk()
    strg.title("set flop(Calling station)")
    strg.geometry("300x200-1300+600")
    strg.resizable(False,False)

    ent1 = Entry(strg, width=15, text="", relief="solid")
    ent1.pack()
    but1 = Button(strg, width=15, text="analysis", overrelief="solid",command=lambda: anal(ent1))
    but1.pack()

    strg.mainloop()

def strategy2():
    strg = Tk()
    strg.title("set flop(Manic)")
    strg.geometry("300x200-1300+600")
    strg.resizable(False,False)

    ent1 = Entry(strg, width=15, text="", relief="solid")
    ent1.pack()
    but1 = Button(strg, width=15, text="analysis", overrelief="solid",command=lambda: anal2(ent1))
    but1.pack()

    strg.mainloop()


def fight1():
    control.traincs(args)

def fight1():
    control.trainma(args)

def quizz1():
    quiz.traincs(args)
def quizz2():
    quiz.trainma(args)
    

    
def select_style3():
    select3 = Tk()
    select3.title("QUIZ")
    select3.geometry("300x200-1300+600")
    select3.resizable(False,False)

    button_cs = Button(select3, width=15, text="Calling Staion", overrelief="solid", command=quizz1)
    button_cs.pack()
    button_ma = Button(select3, width=15, text="Manic", overrelief="solid", command=quizz2)
    button_ma.pack()

    select3.mainloop()




def select_style2():
    select2 = Tk()
    select2.title("VS AI")
    select2.geometry("300x200-1300+600")
    select2.resizable(False,False)

    button_cs = Button(select2, width=15, text="Calling Staion", overrelief="solid", command=fight1)
    button_cs.pack()
    button_ma = Button(select2, width=15, text="Manic", overrelief="solid", command=strategy2)
    button_ma.pack()

    select2.mainloop()

def select_style():
    select1 = Tk()
    select1.title("Poker Learning Tools")
    select1.geometry("300x200-1300+600")
    select1.resizable(False,False)

    button_cs = Button(select1, width=15, text="Calling Staion", overrelief="solid", command=strategy1)
    button_cs.pack()
    button_ma = Button(select1, width=15, text="Manic", overrelief="solid", command=strategy2)
    button_ma.pack()

    select1.mainloop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN/NFSP example in RLCard")
    parser.add_argument(
        '--env',
        type=str,
        default='no-limit-holdem',
        choices=[
            'blackjack',
            'leduc-holdem',
            'limit-holdem',
            'doudizhu',
            'mahjong',
            'no-limit-holdem',
            'uno',
            'gin-rummy',
            'bridge',
        ],
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        default='dqn',
        choices=[
            'dqn',
            'nfsp',
        ],
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=2000,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--log_dircs',
        type=str,
        default='experiments/Exploit_CS4/',
    )
    parser.add_argument(
        '--log_dirma',
        type=str,
        default='experiments/Exploit_MA4/',
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/Advance4/',
    )

    args = parser.parse_args()



root = Tk()

root.title("Poker Learning Tools")
root.geometry("300x200-1300+600")
root.resizable(False,False)

button1 = Button(root, width=15, text="Learn Strategy", overrelief="solid", command=select_style)
button1.pack()
button2 = Button(root, width=15, text="VS AI", overrelief="solid", command=select_style2)
button2.pack()
button3 = Button(root, width=15, text="Quiz", overrelief="solid", command=select_style3)
button3.pack()




root.mainloop()