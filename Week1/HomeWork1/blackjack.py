import numpy as np

#create a box class so that it will be easier to deal with such objects
class Box:
    def __init__(self,bet):
        """
        Create a Box object. A Box is the natural scene of BlackJack, it contains the bet,
        the cards and other information about the game in the Box.
        
        bet: the bet for the game in the Box
        cards: list of integers corresponding to the cards in the Box
        splitted: True of False, whether the Box was splitted / was created by splitting another Box
        doubled: True or False, whether the action 'Double' was played for the Box or not
        bust: True or False, whether the Box has busted or not
        result: float, -1 for lose, +1 for win, +1.5 for natural (BlackJack for the initial cards, without splitting)
        natural: True or False: whether the Box is a natural (BlackJack) or not
        
        result: Box object
        """
        
        self.bet = bet #we have an initial bet for the box
        self.cards = [] #collect the cards here
        self.splitted = False #whether the box is already splitted or not
        self.doubled = False #whether the box is already doubled or not
        self.bust = False #maybe I won't even use this
        self.natural = False #whether the box is a natural or not
        self.left = False #whether the box has already left the game or not
        
        
    def double(self,Deck): 
        """
        the Double action: double the bet and get exactly one more card, 
        then the box finishes its turn
        
        input:
        -Deck: the global deck that is used to draw cards
        
        result: 
        -update the current Box
        """
        
        if self.doubled == False: #if it is not already doubled 
            self.bet = 2.*self.bet #then double the bet
            self.cards.append(Deck.draw()) #draw an additional new card
            self.doubled = True #and set the box to doubled
        else:
            pass #cant double twice
        
    def split(self,Deck): 
        """
        the Split action: the player splits its box if it has the same valued cards,
        the cards will be split among the boxes and both boxes get a new card from the deck
        and the same bet will be used for both of them
        
        input:
        -Deck: global deck used to draw cards for the Boxes
        
        return: 
        -new Box gets created which inherits one of the cards of self and gets the same bet
        """
        
        #create a new box with the same bet
        splitted_box = Box(self.bet)
        
        #give the new box one of the old cards and remove that card from the original box
        splitted_box.cards.append(self.cards[1])
        self.cards.remove(self.cards[1])
        
        #set the boxes to splitted=True
        self.splitted = True
        splitted_box.splitted = True
        
        #draw one card from the deck for each boxes
        self.cards.append(Deck.draw())
        splitted_box.cards.append(Deck.draw())
        
        #here we need to return the new box itself
        return splitted_box
    
    def hit(self,Deck):
        """
        the Hit action: the player asks for another card
        
        input:
        -Deck: global deck used to draw cards for the Boxes

        result: 
        -Box obtains another card
        """
        
        self.cards.append(Deck.draw())
    
    def cardsum(self):
        """
        class method that returns the sum of the cards contained in the Box
        
        return: 
        -integer, the sum of the cards in the Box
        """
        
        cardsum = sum(self.cards) #sum of the cards of the player
        if cardsum > 21 and (11 in self.cards): #if the hand would go bust but the player has an ace it counts as a one 
            j = self.cards.index(11) #so we change that given ace to a one
            self.cards[j] = 1
        #and now return this sum    
        return sum(self.cards)
    
    def double_checker(self):
        """
        class method that tells whether a Box is suitable for a double or not
        
        return:
        -True or False, depending on the cards in the Box
        """
        
        if self.cards[0] == self.cards[1] and len(self.cards) == 2: 
            return True
        else:
            return False
        
        
    def check_natural(self):
        """
        class method that determines whether the box is a natural (blackjack in hand with the 
        first two cards without splitting) or not
        
        return: 
        -True or False, depending on the cards
        """
        
        if self.splitted == True: 
            pass
        else: #if it is not a splitted box
            if len(self.cards) == 2: #if it has exactly two cards
                if self.cards[0]+self.cards[1] == 21: #if the sum of cards is 21
                    self.natural = True
                    return True

        return False
        #Note that in general the player does not win just by having a natural!
    
    def check_doubleace(self):
        """
        class method that determines whether the first two cards are aces or not
        
        return: 
        -True or False
        """
        if len(self.cards) == 2:
            if self.cards[0] == self.cards[1] and self.cards[0] == 11: 
                return True
        return False
        
    def print_box(self):
        """
        class method that prints out general information about the Box
        """
        print("This is a Box")
        print("\t"+"bet: ",self.bet)
        print("\t"+"cards: ",self.cards)
        print("\t"+"natural: ",self.natural)
        print("\t"+"splitted: ",self.splitted)
        print("\t"+"doubled: ",self.doubled)
        print("\t"+"bust: ", self.bust)
        print("\t"+"left: ", self.left)
        
class Deck: #define a class for the deck that is used in the game
    def __init__(self,number_of_packs):
        """
        Create a Deck object. A deck is used to draw cards from it during the game.
        For the whole simulation we need only one deck. Deck objects can be reused.
        
        input: 
        -number_of_packs: the number of packs (52 card french pack) we put into the deck
        
        return:
        -Deck object
        """
        
        s = [2,3,4,5,6,7,8,9,10,10,10,10,11] #according to the rule of blackjack these are the possible values of cards
        all_packs = []
        for i in range(number_of_packs):
            all_packs.append([s,s,s,s]) #put in exactly one french pack

        all_packs = np.array(all_packs).reshape(len(s)*4*number_of_packs) #contains all the cards that we have in the deck

        #with this we have added all the packs to the deck
        #now we have to shuffle it
        #for this shuffling we just use this simple built in function of numpy
        #this is because we will shuffle the deck anyways before the game
        self.deck = np.random.choice(all_packs,size = len(all_packs), replace = False)
        #for the other shufflings we will use something else
        
        self.cards_drawn = 0 #number of cards drawn from the deck
        
        #general info about the deck itself
        self.nof_cards = number_of_packs*52
        self.nof_packs = number_of_packs
        
    def draw(self):
        """
        Deck method, draw a card from the top of the deck
        
        return:
        -a card from the top of the deck
        """
        
        #we draw the next card from the deck
        next_card = self.deck[self.cards_drawn]
        
        #and set the counter to +1 so next time we will draw the next card
        self.cards_drawn += 1 
        return next_card
    
    def shuffle(self):
        """
        Deck method; shuffle the cards of the deck using a specific shuffling method
        
        return: 
        -freshly shuffled deck
        """
        
        n = self.nof_cards #get the size of the deck
        new_deck = self.deck.copy() #copy the current deck
        
        for i in range(n): #iterate through the deck
            j = i + np.random.randint(n-i) #for every card indexed by i choose another random card with i<=j 
            temporary = new_deck[i] #exchange cards with index i and j
            new_deck[i] = new_deck[j]
            new_deck[j] = temporary
        
        #put back the freshly shuffled deck
        self.deck = new_deck
        #and set the cards_drawn to zero as after a shuffle we always start a new game
        self.cards_drawn = 0
        
    def print_deck(self):
        """
        Deck method; print useful information about the Deck object
        """
        #maybe add a deck printing option for debugging
        print("This is a Deck")
        print("\t"+"deck: ",self.deck)
        print("\t"+"cards drawn: ",self.cards_drawn)
        print("\t"+"number of packs :",self.nof_packs)
        print("\t"+"number of cards :",self.nof_cards)
        
def create_game(global_DECK,nof_players):
    """
    Create a game of BlackJack. That is, give a card to the dealer, put bet into the Boxes, give two cards to the Boxes
    
    input:
    -global_DECK: global Deck object
    -nof_players: integer, the number of players playing the game
    
    return:
    -boxes: a list of Box objects, ready for the player to carry out actions according to its strategy
    -dealer_card: the card of the dealer that is visible to all the players
    """
    
    #let's start by shuffling the deck
    global_DECK.shuffle()
    
    #give the dealer one card
    dealer_card = global_DECK.draw()
    
    #now create the Boxes for the players and give them the cards
    #and then give two-two cards for each box
    boxes = []
    for i in range(nof_players):
        boxes.append(Box(bet = 10.)) #the basic bet is 10 units of money
        boxes[i].cards = [global_DECK.draw(),global_DECK.draw()] #give two cards to the Box
    
    return [boxes,dealer_card]


def strategy1(box,boxes,global_DECK,dealer_card,j,splits):
    #split if possible    
    while box.double_checker() and splits < 5:
        new_box = box.split(global_DECK)
        
        #insert the new box TO THE RIGHT of the original one (I think this is also important)
        boxes.insert(j+1,new_box)
        splits += 1
        #now if it happens that the cards at box are again doubles then the player can split again and the 
        #new splitted deck will get put between the original deck and the previous splitted deck
    
    #and now we continue with the original box and later on go back to the splitted box(es)
    bsum = box.cardsum()
    #now let's just forget about the dealers_card
    if bsum > 11:
        return #dont do anything,essentially wait for the dealer to bust
    elif bsum <= 11: #if it has low sum then hit
        box.hit(global_DECK)
        return
    
def strategy2(box,boxes,global_DECK,dealer_card,j,splits):
    #always split if possible
    while box.double_checker() and splits < 5:
        new_box = box.split(global_DECK)
        boxes.insert(j+1,new_box)
        splits += 1
    bsum = box.cardsum()   
    #now also look at the dealer's card
    if bsum > 11 and dealer_card < 7:
        return #dont do anything, wait for the dealer to bust
    
    elif bsum <= 11: #if it has low sum then double 
        box.double(global_DECK)
        return 
    
    #otherwise while bsum is between 11 and 17 hit 
    while box.cardsum() > 11 and box.cardsum()<17 and dealer_card >= 7:
        box.hit(global_DECK)
    return

def strategy3(box,boxes,global_DECK,dealer_card,j,splits):
    #split every pair above 6
    while box.double_checker() and box.cardsum() >= 7 and splits < 5:
        new_box = box.split(global_DECK)
        
        boxes.insert(j+1,new_box)
        splits += 1
    
    bsum = box.cardsum()
    if bsum > 11 and dealer_card < 7:
        return #dont do anything
    
    elif bsum <= 11: #if it has low sum then double 
        box.double(global_DECK)
        return
    
    #now if bsum is above 11 and 14 maybe it will be better
    while box.cardsum() > 11 and box.cardsum()<14 and dealer_card >= 7:
        box.hit(global_DECK)
    return

def strategy4(box,boxes,global_DECK,dealer_card,j,splits):
    #split every pair above 6 but only if the dealer has lower than 8
    while box.double_checker() and box.cardsum() >= 7 and splits < 5 and dealer_card < 8:
        new_box = box.split(global_DECK)
        
        boxes.insert(j+1,new_box)
        splits += 1
    
    #same as previously
    bsum = box.cardsum()
    if bsum > 11 and dealer_card < 7:
        return #dont do anything
    elif bsum <= 11: #if it has low sum then double 
        box.double(global_DECK)
        return
    
    #now if bsum is above 11 and 14 maybe it will be better
    while box.cardsum() > 11 and box.cardsum()<14 and dealer_card >= 7:
        box.hit(global_DECK)   
    return

def strategy5(box,boxes,global_DECK,dealer_card,j,splits):
    #always split with two aces, and also always split two 9s and two 8s but never ever split a ten!!!
    while box.double_checker() and (box.check_doubleace() or box.cardsum() == 16 or box.cardsum() == 18) and splits < 5:
        new_box = box.split(global_DECK)
        boxes.insert(j+1,new_box)
        splits += 1
        
    #split 7,6,3,2 depending on the dealers card #but only if the dealers card is smaller than 8
    while box.double_checker() and (box.cardsum() == 14 or box.cardsum() == 12 or box.cardsum() == 6 or box.cardsum() == 4) and splits < 5 and dealer_card<8:
        new_box = box.split(global_DECK)
        boxes.insert(j+1,new_box)
        splits += 1
    
    bsum = box.cardsum()
    
    if bsum >= 17: #always Stand with 17 or above
        return
    elif bsum < 17 and bsum > 11 and dealer_card < 7: #stand here as well
        return 
    elif bsum < 17 and bsum > 11 and dealer_card >= 7: #hit here
        box.hit(global_DECK) 
        return
    elif bsum == 11: #always double here
        box.double(global_DECK)
        return 
    elif bsum == 10 and dealer_card < 10: #also double here
        box.double(global_DECK)
        return 
    elif bsum == 10 and dealer_card >= 10: #only hit here
        box.hit(global_DECK) 
        return        
    elif bsum == 9 and dealer_card < 7: 
        box.double(global_DECK)
        return 
    elif bsum == 9 and dealer_card >= 7: 
        box.hit(global_DECK) 
        return          
    elif bsum < 9:
        box.hit(global_DECK) 
        return 
    
    
def dealer_card_sum(dealer_cards):
    """
    function that counts the total value of the cards of the dealer 
    
    input: 
    -list of the dealer's cards
    
    return: 
    -integer, the total sum of the cards (taking into account
    the fact that aces can be either 11 or 1)
    """
    cardsum = sum(dealer_cards) #sum of the cards of the dealer as they are
    if cardsum > 21 and (11 in dealer_cards): #if the hand would go bust but the dealer has an ace it counts as a one 
        j = dealer_cards.index(11) #so we change that given ace to a one
        dealer_cards[j] = 1
    #and now return this sum
    return sum(dealer_cards)

def remaining_boxes(boxes):#collect the boxes which did not already leave the deck and still have to evaluate their cards
    """
    function that counts the Boxes which are still present at the desk.
    That is, when the game ends, the players win/lose their bet (or simply just get it back) and then leave the desk.
    This function is useful when one starts to evaluate the different types of Boxes
    
    input: 
    -list of Box objects
    
    return: 
    -list of integers, representing the indices of those Box objects in the 'boxes' list, whose are still present at the desk
    """
    rb = []
    for idx,box in enumerate(boxes): #iterate through the boxes
        if box.left == False: #if the box did not leave already
            rb.append(idx) #add its index to the list
            
    return rb

def end_of_game(boxes, dealer_cards):
    """
    function that evaluates the cards at the end of the game
    
    input: 
    -boxes, dealer_cards
    
    return: 
    -total income for the Boxes
    """
    tot_income = 0 #we count the total income 
    
    #first we must get rid of the naturals because they get their reward immediately
    for idx,box in enumerate(boxes):
        if box.natural == True: 
            tot_income += box.bet*1.5
            box.left = True #the box then leaves
            
        elif box.bust == True: #else if the box went bust then the player loses the bet
            tot_income -= box.bet
            box.left = True #the box leaves
    #now only those boxes are left which are not naturals or busted
    
    #sum of the dealers cards
    dcs = dealer_card_sum(dealer_cards)
    #if the dealer goes bust every box wins its bet if the box is not bust or natural 
    #but we need to iterate throug those boxes which are still present and not left
    rb = remaining_boxes(boxes) #index of the remaining boxes
    
    if dcs > 21: 
        for idx in rb:
            box = boxes[idx]
            tot_income += box.bet
            box.left = True #the box leaves the desk
            
    rb = remaining_boxes(boxes)  #now only these boxes remain
    #actually if there are still boxes left then this list should be the same as the previous
    #now we have to evaluate the condition where non of the box and the dealer goes bust and we have to actually compare the cards
    for idx in rb:
        box = boxes[idx]
        if dcs == box.cardsum(): #have the same cards
            #nothing happens, the box leaves
            box.left = True
        elif dcs > box.cardsum(): #the dealer has higher
            tot_income -= box.bet 
            box.left = True
        else: #the box has higher cards
            tot_income += box.bet
            box.left = True
            
    #at this point there should be no other boxes left 
    rb = remaining_boxes(boxes)
    if rb != []:
        print("Error! There should be no boxes left at this point!")

    return tot_income

def play_game(boxes, global_DECK, dealer_card, strategy):
    """
    function that simulates the game of Black Jack
    the players play according to a given strategy
    
    inputs:
    -boxes: list of Box objects
    -global_DECK: Deck object 
    -dealer_card: integer, card of the dealer
    -strategy: function, a well defined behaviour of the player
    
    output: 
    -float, the total income of the whole game 
    """
    #now each player plays the game for each box we have
    #this means that we have to iterate through the boxes but it is a list which can grow, for example by splitting a box
    
    #number of times the players splits
    #let's say that the player can only split 5 times in a game
    splits = 0
    
    j = 0
    while j<len(boxes):
        
        if boxes[j].check_natural(): #if it is a natural we don't have to do anything, it is a win anyways 
            j+=1 #just skip through
            
        else: #else we need to use a strategy
            box = boxes[j]
            strategy(box,boxes,global_DECK,dealer_card,j,splits)
            
            if box.cardsum() > 21: 
                box.bust = True
                
            j+=1
    
    #now the dealer draws cards  
    dealer_cards = [dealer_card]
    
    while dealer_card_sum(dealer_cards) < 17: #if the dealer has lower than 17 it must draw another card
        dealer_cards.append(global_DECK.draw()) #it draws a card from the deck
        #If an ace would bust the dealers hand than it must be counted as a one from that point
    #now everybody finished its turn, only thing left is to 
    
    tot_income = end_of_game(boxes, dealer_cards)
    
    return tot_income

def incomestatistics(global_DECK, iterations, strategy):
    """
    function that carries out several instances of the Black Jack simulation
    
    inputs:
    -global_DECK: global Deck object
    -iterations: integer, the number of simulations to be carried out
    -strategy: function, the well defined behaviour of the player
    
    outputs:
    -float, the average net income per round of Black Jack
    -list of floats, corresponding to the income for each simulation
    """
    final_income = 0 #count the income for all the iterations
    data = []
    for i in range(iterations):
        [boxes, dealer_card] = create_game(global_DECK = global_DECK, nof_players = 1) #create the new game with boxes and dealer card
        income = play_game(boxes = boxes, global_DECK = global_DECK, dealer_card = dealer_card, strategy = strategy) #play the game 
        final_income += income 
        data.append(income)
    return final_income/iterations, data