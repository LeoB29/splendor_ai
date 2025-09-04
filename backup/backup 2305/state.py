from utils import *

from utils import np

import numpy as np



params = {
    "number_cards_board": 12,
    "number_features_card" : 8,
    "initial_card_deck" : [
        ## diamonds
        [0, 0, 0, 0, 2, 1, 0, 2],
        [0, 0, 0, 0, 0, 1, 2, 0],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 3, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 2],
        [0, 0, 0, 0, 1, 1, 1, 2],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 4],
        [1, 1, 0, 0, 0, 2, 2, 3],
        [1, 1, 0, 2, 3, 0, 3, 4],
        [1, 2, 0, 0, 0, 2, 4, 1],
        [1, 2, 0, 0, 0, 0, 5, 4],
        [1, 2, 0, 0, 0, 3, 5, 4],
        [1, 3, 0, 6, 0, 0, 0, 4],
        [2, 3, 0, 0, 3, 3, 5, 3],
        [2, 4, 0, 0, 0, 7, 0, 0],
        [2, 4, 0, 3, 0, 6, 3, 0],
        [2, 5, 0, 3, 0, 7, 0, 0],
        ## sapphires
        [0, 0, 1, 1, 0, 2, 0, 0],
        [0, 0, 1, 1, 0, 1, 2, 1],
        [0, 0, 1, 1, 0, 1, 1, 1],
        [0, 0, 1, 0, 1, 0, 1, 3],
        [0, 0, 1, 0, 0, 3, 0, 0],
        [0, 0, 1, 1, 0, 0, 2, 2],
        [0, 0, 1, 0, 0, 2, 0, 2],
        [0, 1, 1, 0, 0, 0, 4, 0],
        [1, 1, 1, 0, 2, 0, 3, 2],
        [1, 1, 1, 0, 2, 3, 0, 3],
        [1, 2, 1, 5, 3, 0, 0, 0],
        [1, 2, 1, 0, 5, 0, 0, 0],
        [1, 2, 1, 2, 0, 4, 1, 0],
        [1, 3, 1, 0, 6, 0, 0, 0],
        [2, 3, 1, 3, 0, 5, 3, 3],
        [2, 4, 1, 7, 0, 0, 0, 0],
        [2, 4, 1, 6, 3, 3, 0, 0],
        [2, 5, 1, 7, 3, 0, 0, 0],
        ## obsidians
        [0, 0, 2, 1, 1, 0, 1, 1],
        [0, 0, 2, 0, 0, 0, 1, 2],
        [0, 0, 2, 2, 0, 0, 0, 2],
        [0, 0, 2, 0, 0, 1, 3, 1],
        [0, 0, 2, 0, 0, 0, 0, 3],
        [0, 0, 2, 1, 2, 0, 1, 1],
        [0, 0, 2, 2, 2, 0, 1, 0],
        [0, 1, 2, 0, 4, 0, 0, 0],
        [1, 1, 2, 3, 2, 0, 0, 2],
        [1, 1, 2, 3, 0, 2, 0, 3],
        [1, 2, 2, 0, 1, 0, 2, 4],
        [1, 2, 2, 5, 0, 0, 0, 0],
        [1, 2, 2, 0, 0, 0, 3, 5],
        [1, 3, 2, 0, 0, 6, 0, 0],
        [2, 3, 2, 3, 3, 0, 3, 5],
        [2, 4, 2, 0, 0, 0, 7, 0],
        [2, 4, 2, 0, 0, 3, 6, 3],
        [2, 5, 2, 0, 0, 3, 7, 0],
        ## rubies
        [0, 0, 3, 3, 0, 0, 0, 0],
        [0, 0, 3, 1, 0, 3, 1, 0],
        [0, 0, 3, 0, 2, 0, 0, 1],
        [0, 0, 3, 2, 0, 2, 0, 1],
        [0, 0, 3, 2, 1, 1, 0, 1],
        [0, 0, 3, 1, 1, 1, 0, 1],
        [0, 0, 3, 2, 0, 0, 2, 0],
        [0, 1, 3, 4, 0, 0, 0, 0],
        [1, 1, 3, 0, 3, 3, 2, 0],
        [1, 1, 3, 2, 0, 3, 2, 0],
        [1, 2, 3, 1, 4, 0, 0, 2],
        [1, 2, 3, 3, 0, 5, 0, 0],
        [1, 2, 3, 0, 0, 5, 0, 0],
        [1, 3, 3, 0, 0, 0, 6, 0],
        [2, 3, 3, 3, 5, 3, 0, 3],
        [2, 4, 3, 0, 0, 0, 0, 7],
        [2, 4, 3, 0, 3, 0, 3, 6],
        [2, 5, 3, 0, 0, 0, 3, 7],
        ## emeralds
        [0, 0, 4, 2, 1, 0, 0, 0],
        [0, 0, 4, 0, 2, 0, 2, 0],
        [0, 0, 4, 1, 3, 0, 0, 1],
        [0, 0, 4, 1, 1, 1, 1, 0],
        [0, 0, 4, 1, 1, 2, 1, 0],
        [0, 0, 4, 0, 1, 2, 2, 0],
        [0, 0, 4, 0, 0, 0, 3, 0],
        [0, 1, 4, 0, 0, 4, 0, 0],
        [1, 1, 4, 3, 0, 0, 3, 2],
        [1, 1, 4, 2, 3, 2, 0, 0],
        [1, 2, 4, 4, 2, 1, 0, 0],
        [1, 2, 4, 0, 0, 0, 0, 5],
        [1, 2, 4, 0, 5, 0, 0, 3],
        [1, 3, 4, 0, 0, 0, 0, 6],
        [2, 3, 4, 5, 3, 3, 3, 0],
        [2, 4, 4, 3, 6, 0, 0, 3],
        [2, 4, 4, 0, 7, 0, 0, 0],
        [2, 5, 4, 0, 7, 0, 0, 3]
    ],
    "initial_noble_deck" : [
        [0, 3, 0, 3, 3],
        [3, 3, 3, 0, 0],
        [4, 0, 4, 0, 0],
        [4, 4, 0, 0, 0],
        [0, 4, 0, 0, 4],
        [3, 3, 0, 0, 3],
        [3, 0, 3, 3, 0],
        [0, 0, 3, 3, 3],
        [0, 0, 4, 4, 0],
        [0, 0, 0, 4, 4]
        ]}   


### generate initial board

def generate_board():

#cards_table = np.zeros((params["number_cards_board"],params["number_features_card"]))
# initial board
    current_deck = params["initial_card_deck"]
    cards_tier0, current_deck = draw_cards(4,0,current_deck)
    cards_tier1, current_deck = draw_cards(4,1,current_deck)
    cards_tier2, current_deck = draw_cards(4,2,current_deck)

    cards_on_board = cards_tier0 + cards_tier1 + cards_tier2

    nobles_on_board = random.sample(params["initial_noble_deck"], 3)

    gem_bank = [4,4,4,4,4,5]

    # player resources
    player_points = [0,0]

    #gems_player1 = [0,0,0,0,0,0]
    #gems_player2 = [0,0,0,0,0,0]

    player_gems = [
        [0,0,0,0,0,0],
        [0,0,0,0,0,0]
    ]

    #cards_built_player1 = np.zeros((0,params["number_features_card"]))
    #cards_built_player2= np.zeros((0,params["number_features_card"]))
    cards_built = np.zeros((0,params["number_features_card"]+1))

    cards_in_hand = np.zeros((0,params["number_features_card"]+1))
    #cards_hand_player1 = np.zeros((0,params["number_features_card"]))
    #cards_hand_player2 = np.zeros((0,params["number_features_card"]))

    return(cards_on_board,nobles_on_board,gem_bank,player_points,player_gems,cards_built,cards_in_hand,current_deck)

def draw_cards(number_cards, tier, card_deck):
    card_deck_tiered = [row for row in card_deck if row[0] == tier]
    cards = random.sample(card_deck_tiered, number_cards)
    card_deck = [row for row in card_deck if row not in cards]
    return(cards,card_deck)


### possible actions

### player_number should be 1 or 2 
# get gems - color combination arg should be an array with the number corresponding to the colour [0,0] for two white [1,2,3] for blue/black/red   
def get_gems(color_combination,gem_bank,player_gems,player_number):
    for i, color in enumerate(color_combination):
        gem_bank[color] -= 1
        player_gems[player_number-1][color] += 1

    return(gem_bank,player_gems)


# card index arg should be a number between 0 and 11
def book_card(card_index,cards_on_board,cards_in_hand,current_deck,gem_bank,player_gems,player_number):
    
    # put new card in player hand
    selected_card = cards_on_board.pop(card_index)
    selected_card = np.append(selected_card, player_number)
    selected_card_tier = selected_card[0]
    cards_in_hand = np.append(cards_in_hand,selected_card)
    
    # replace card on board
    new_card, current_deck = draw_cards(1,selected_card_tier,current_deck)
    cards_on_board =  cards_on_board + new_card

    # give gold to player and remove from bank
    if gem_bank[5] > 0:
        player_gems[player_number-1][5] += 1
        gem_bank[5] -= 1

    return(cards_in_hand,cards_on_board,current_deck,player_gems,gem_bank)


test, test2 = book_card(card_index,cards_on_board,cards_in_hand,gem_bank,player_gems,1)



# card index arg should be a number between  and 11 (if building a card on board) or between 12 and 14 (if building a card in hand) // use_gold a value between 0 and 3 (number of gold tokens used) // goldX_color a number between 0 and 5 corresponding to the stone
def build_card(card_index,cards_on_board,cards_in_hand,cards_built,gem_bank,player_gems,current_deck,player_number,use_gold=0,gold1_color=None,gold2_color=None,gold3_color=None):
    if card_index<12:
        selected_card = cards_on_board.pop(card_index)
        selected_card_tier = selected_card[0]
        selected_card = np.append(selected_card, player_number)
        new_card, current_deck = draw_cards(1,selected_card_tier, current_deck)
        cards_on_board =  cards_on_board + new_card
    elif card_index>=12:
        selected_card = cards_in_hand.pop(card_index-12)
    
    cards_built = np.append(cards_built,selected_card)

    #### add gem payment
    # pay gold
    player_gems[player_number -1][5] = player_gems[player_number -1][5] - use_gold
    gem_bank[5] += use_gold

    if gold1_color is not None:
        selected_card[3+gold1_color] -=1
    if gold2_color is not None:
        selected_card[3+gold2_color] -=1
    if gold3_color is not None:
        selected_card[3+gold3_color] -= 1
    
    # pay the card 
    for i in range(0, 5):
        player_gems[player_number - 1][i] = player_gems[player_number - 1][i] - selected_card[3+i]
        gem_bank[i] = gem_bank[i] - selected_card[3+i]

    return(cards_on_board,cards_in_hand,cards_built,gem_bank,player_gems,current_deck)

        

cards_on_board,nobles_on_board,gem_bank,player_points,player_gems,cards_built,cards_in_hand,current_deck = generate_board()

cards_on_board,cards_in_hand,cards_built,gem_bank,player_gems,current_deck = build_card(3,cards_on_board,cards_in_hand,cards_built,gem_bank,player_gems,current_deck,1)




cards_on_board,nobles_on_board,gem_bank,player_points,player_gems,cards_built,cards_in_hand,current_deck = generate_board()








def check_state():





#features by card:
#1 rang 1 2 3 
#2 points
#3 couleur
#4 prix blanc
#5 prix bleu 
#6 prix noir
#7 prix rouge
#8 prix vert
#blanc : 0
#bleu : 1
#noir : 2
#rouge : 3
#vert : 4



card_deck = [
    ## diamonds
    [0, 0, 0, 0, 2, 1, 0, 2],
    [0, 0, 0, 0, 0, 1, 2, 0],
    [0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 2],
    [0, 0, 0, 0, 1, 1, 1, 2],
    [0, 0, 0, 0, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 4],
    [1, 1, 0, 0, 0, 2, 2, 3],
    [1, 1, 0, 2, 3, 0, 3, 4],
    [1, 2, 0, 0, 0, 2, 4, 1],
    [1, 2, 0, 0, 0, 0, 5, 4],
    [1, 2, 0, 0, 0, 3, 5, 4],
    [1, 3, 0, 6, 0, 0, 0, 4],
    [2, 3, 0, 0, 3, 3, 5, 3],
    [2, 4, 0, 0, 0, 7, 0, 0],
    [2, 4, 0, 3, 0, 6, 3, 0],
    [2, 5, 0, 3, 0, 7, 0, 0],
    ## sapphires
    [0, 0, 1, 1, 0, 2, 0, 0],
    [0, 0, 1, 1, 0, 1, 2, 1],
    [0, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 1, 0, 1, 0, 1, 3],
    [0, 0, 1, 0, 0, 3, 0, 0],
    [0, 0, 1, 1, 0, 0, 2, 2],
    [0, 0, 1, 0, 0, 2, 0, 2],
    [0, 1, 1, 0, 0, 0, 4, 0],
    [1, 1, 1, 0, 2, 0, 3, 2],
    [1, 1, 1, 0, 2, 3, 0, 3],
    [1, 2, 1, 5, 3, 0, 0, 0],
    [1, 2, 1, 0, 5, 0, 0, 0],
    [1, 2, 1, 2, 0, 4, 1, 0],
    [1, 3, 1, 0, 6, 0, 0, 0],
    [2, 3, 1, 3, 0, 5, 3, 3],
    [2, 4, 1, 7, 0, 0, 0, 0],
    [2, 4, 1, 6, 3, 3, 0, 0],
    [2, 5, 1, 7, 3, 0, 0, 0],
    ## obsidians
    [0, 0, 2, 1, 1, 0, 1, 1],
    [0, 0, 2, 0, 0, 0, 1, 2],
    [0, 0, 2, 2, 0, 0, 0, 2],
    [0, 0, 2, 0, 0, 1, 3, 1],
    [0, 0, 2, 0, 0, 0, 0, 3],
    [0, 0, 2, 1, 2, 0, 1, 1],
    [0, 0, 2, 2, 2, 0, 1, 0],
    [0, 1, 2, 0, 4, 0, 0, 0],
    [1, 1, 2, 3, 2, 0, 0, 2],
    [1, 1, 2, 3, 0, 2, 0, 3],
    [1, 2, 2, 0, 1, 0, 2, 4],
    [1, 2, 2, 5, 0, 0, 0, 0],
    [1, 2, 2, 0, 0, 0, 3, 5],
    [1, 3, 2, 0, 0, 6, 0, 0],
    [2, 3, 2, 3, 3, 0, 3, 5],
    [2, 4, 2, 0, 0, 0, 7, 0],
    [2, 4, 2, 0, 0, 3, 6, 3],
    [2, 5, 2, 0, 0, 3, 7, 0],
    ## rubies
    [0, 0, 3, 3, 0, 0, 0, 0],
    [0, 0, 3, 1, 0, 3, 1, 0],
    [0, 0, 3, 0, 2, 0, 0, 1],
    [0, 0, 3, 2, 0, 2, 0, 1],
    [0, 0, 3, 2, 1, 1, 0, 1],
    [0, 0, 3, 1, 1, 1, 0, 1],
    [0, 0, 3, 2, 0, 0, 2, 0],
    [0, 1, 3, 4, 0, 0, 0, 0],
    [1, 1, 3, 0, 3, 3, 2, 0],
    [1, 1, 3, 2, 0, 3, 2, 0],
    [1, 2, 3, 1, 4, 0, 0, 2],
    [1, 2, 3, 3, 0, 5, 0, 0],
    [1, 2, 3, 0, 0, 5, 0, 0],
    [1, 3, 3, 0, 0, 0, 6, 0],
    [2, 3, 3, 3, 5, 3, 0, 3],
    [2, 4, 3, 0, 0, 0, 0, 7],
    [2, 4, 3, 0, 3, 0, 3, 6],
    [2, 5, 3, 0, 0, 0, 3, 7],
    ## emeralds
    [0, 0, 4, 2, 1, 0, 0, 0],
    [0, 0, 4, 0, 2, 0, 2, 0],
    [0, 0, 4, 1, 3, 0, 0, 1],
    [0, 0, 4, 1, 1, 1, 1, 0],
    [0, 0, 4, 1, 1, 2, 1, 0],
    [0, 0, 4, 0, 1, 2, 2, 0],
    [0, 0, 4, 0, 0, 0, 3, 0],
    [0, 1, 4, 0, 0, 4, 0, 0],
    [1, 1, 4, 3, 0, 0, 3, 2],
    [1, 1, 4, 2, 3, 2, 0, 0],
    [1, 2, 4, 4, 2, 1, 0, 0],
    [1, 2, 4, 0, 0, 0, 0, 5],
    [1, 2, 4, 0, 5, 0, 0, 3],
    [1, 3, 4, 0, 0, 0, 0, 6],
    [2, 3, 4, 5, 3, 3, 3, 0],
    [2, 4, 4, 3, 6, 0, 0, 3],
    [2, 4, 4, 0, 7, 0, 0, 0],
    [2, 5, 4, 0, 7, 0, 0, 3]
]  



noble_deck = [
    [0, 3, 0, 3, 3],
    [3, 3, 3, 0, 0],
    [4, 0, 4, 0, 0],
    [4, 4, 0, 0, 0],
    [0, 4, 0, 0, 4],
    [3, 3, 0, 0, 3],
    [3, 0, 3, 3, 0],
    [0, 0, 3, 3, 3],
    [0, 0, 4, 4, 0],
    [0, 0, 0, 4, 4]
]