import random

def number_guessing_game():
    number = random.randint(1, 100)
    guess = 0
    while guess != number:
        guess = int(input("Guess a number between 1 and 100: "))
        if guess < number:
            print("Too low!")
        elif guess > number:
            print("Too high!")
        else:
            print("You guessed it! The number was", number)

number_guessing_game()