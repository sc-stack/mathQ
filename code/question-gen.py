import json
import tiktoken
import numpy as np
import openai
import os
from collections import defaultdict

openai.api_key = "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"

with open("test.json", "r") as f:
    dataDict = json.load(f)
    newString = json.dumps(dataDict, indent=4)

heart_of_algebra_categories = ["Solving linear equations and linear inequalities", "Interpreting linear functions", "Linear Equation Word Problems", "Linear Inequality Word Problems", 
            "Graphing Linear Equations", "Linear Function Word Problems", "Systems of linear inequalities word problems", "Solving Systems of Linear Equations", "System of Linear Equations Word Problems"]

passport_to_advanced_math_categories = ["Solving Quadratic Equations", "Interpreting nonlinear expressions", "Quadratic and Exponential Word Problems", "Manipulating quadratic and exponential expressions", "Radical and rational expressions", "Radical and rational equations", "Operations with rational expressions", "Polynomial Factors and Graphs", "Non-Linear Equation Graphs", "Linear and Quadratic Systems", "Structure in expressions", "Isolating Quantities", "Function Notation"]

format = "multiple-choice"

### Heart of algebra

# Solving linear equations and linear inequalities (Open Ended)
# ChatGPT GPT-4 Success Rate For Khan Academy Questions 5/5
# GPT-4 Success Rate For Generated Questions 
sample_questions_0 = ['j/2 + 7 = 12. What is the value of j in the equation shown above?', 'k/4 + 3 = 14. What is the value of k in the equation shown above?', 
                    '6 = 2(y+2). What is the value of y in the equation shown above?', '43 = 8c - 5. What is the value of c in the equation shown above?', '41 = 12d - 7. What is the value of d in the equation shown above?']

# Interpreting linear functions (Open Ended)
# ChatGPT GPT-4 Success Rate For Khan Academy Questions 5/5
# GPT-4 Success Rate For Generated Questions 
sample_questions_1 = ['D = -200t + 9000. Harry took a loan from the bank. Each month, he pays a fixed amount of money back to the bank. The equation above shows the remaining amount of the loan, D, measured in dollars, after t months. How much does Harry pay back to the bank each month, in dollars?',
                    "W = 32 - 0.05n; Andrei has a glass tank. First, he wants to put some marbles in it, all of the same volume. Then, he wants to fill the tank with water until it's completely full. The equation shown above describes the volume of water, W, measured in liters, that Andrei should use when there are n marbles. What is the volume of the glass tank, in liters?",
                    "R = 42 - 0.7t; Quinn returned home one summer's day to find it extremely hot. He turned the air conditioner on, and the room's temperature began decreasing at a constant rate. The equation shown above gives the room's temperature, R, in degrees Celsius, t minutes after Quinn turned on the air conditioner. What was the room's temperature, in degrees Celsius, when Quinn returned home?",
                    "D = -38t + 220; Rachel is driving a race car at a constant speed on a closed course. The formula shown above describes the remaining distance, D, measured in meters, that Rachel has to travel after t seconds. What is Rachel's speed in meters per second?"
                    "S = 40,000 + 500c; Caden started a new job selling dental chairs. He earns a base salary plus a commission for every chair he sells. The equation above gives Caden's annual salary, S, in dollars, after selling c dental chairs. Based on the equation above, what is Caden's base salary?"
]

# Linear Equation Word Problems (Multiple Choice)
# ChatGPT GPT-4 Success Rate For Khan Academy Questions 5/5
# GPT-4 Success Rate For Generated Questions 
sample_questions_2 = [
    "John read the first 114 pages of a novel, which was 3 pages less than 1/3 of the novel. If p is the total number of pages in the novel, which of the following equations best describes the situation? Choose 1 answer: (a) (1/3)p - 3 = 114, (b) (1/3)p + 3 = 114, (c) 3p - 3 = 114, (d) 3p + 3 = 114",
    "Jack's mother gave him 50 chocolates to give to his friends at his birthday party. He gave 3 chocolates to each of his friends and still had 2 chocolates left. If x is the number of friends at Jack's party, which of the following equations best describes the situation? Choose 1 answer: (a) 3x + 50 = 2, (b) 2x + 3 = 50, (c) 3x + 2 = 50, (d) 3x - 2 = 50",
    "Felipe is saving money for a class trip. He already has saved $250 that he will put toward the trip. To save more money for the trip, Felipe gets a job where each month he can add $350 to his savings for the trip. Let m be the number of months that Felipe has worked at his new job. If Felipe needs to save $2700 to go on the trip, which equation best models the situation? Choose 1 answer: (A) 250m - 350 = 2700, (B) 250m + 350 = 2700, (C) 350m - 250 = 2700, (D) 350m + 250 = 2700",
    "A pizza delivery worker purchased a used motor scooter that had been driven 12,100 miles. He drives the motor scooter only on days he is working, during which he drives an average of 50 miles per day. After d days of pizza delivery, the motor scooter has been driven a total of 25,000 miles. Which of the following equations best models this situation? Choose 1 answer: (A) 50(12,100 + d) = 25,000, (B) 12,100 + (d/50) = 25,000, (C) 12,100d + 50 = 25,000, (D) 12,100 + 50d = 25,000",
    "An elite runner's stride rate is currently 168 strides per minute (the number of steps she takes per minute). She wants to improve her stride rate to 180, and believes that through training, she can increase her stride rate by 2 strides per minute each day, d. Which equation best models the situation? Choose 1 answer: (A) 168 - 2d = 180, (B) 168 + 2d = 180, (C) 180 - 120d = 168, (D) 180 + 120d = 168",
]

# Linear Inequality Word Problems (Multiple Choice)
# ChatGPT GPT-4 Success Rate For Khan Academy Questions 5/5
# GPT-4 Success Rate For Generated Questions 
sample_questions_3 = [
    "Julia has $5.00 to spend on lemons. Lemons cost $0.59 each, and there is no tax on the purchase. Which of the following inequalities can be used to represent x, the number of lemons Julia can buy? Choose 1 answer: (A) (x/0.59) ≥ 5, (B) (x/0.59) ≤ 5, (C) 0.59x ≥ 5, (D) 0.59x ≤ 5",
    "Dominique is allowed to play up to 8 hours of video games this week. They want to play video games for at least 4 hours this weekend. Which of the following can be used to represent t, the number of hours they can play video games before the weekend? Choose 1 answer: (A) 8 - t ≥ 4, (B) 8 - t ≤ 4, (C) t - 8 ≥ 4, (D) t - 8 ≤ 4",
    "Dorothy is making sausages. If she wants to make at least 24 sausages that weigh 12 ounces each, which of the following best describes x, the total weight of the sausages in ounces? Choose 1 answer: (A) x/12 ≥ 24, (B) x/12 ≤ 24, (C) x - 12 ≥ 24, (D) x - 12 ≤ 24",
    "Anton must finish reading a 369-page book in the next 7 days. Which of the following inequalities can be used to represent x, the average number of pages he must read each day to finish the book on or ahead of schedule? Choose 1 answer: (A) x/7 ≥ 369, (B) x + 7 ≥ 369, (C) 7x ≥ 369, (D) 7x < 369",
    "Victoria is running a fundraising campaign for the local community center. Her goal is to raise more than $5,500. If the campaign has already raised $2,324, which of the following inequalities can be solved to find x, the amount of money that still needs to be raised for Victoria to meet her goal? Choose 1 answer: (A) x - 2,324 > 5,500, (B) 2,324 + x > 5,500, (C) x - 2,324 < 5,500, (D) 2,324 + x < 5,500"
]

# Graphing Linear Equations (Open Ended) (May need to generate graph images)
# ChatGPT GPT-4 Success Rate For Khan Academy Questions 5/5
# GPT-4 Success Rate For Generated Questions 
sample_questions_4 = [
    "What is the slope of the line represented by the equation 3x - 4y = 7?",
    "What is the slope of the line represented by the equation 2x - 5y = 9?",
    "The equation 8x - 6y = 1 is graphed in the xy-plane. What is the slope of the line?",
    "What is the slope of the line represented by the equation x - 3y = 10?",
    "The equation 2y - 7x = 5 is graphed in the xy-plane. What is the slope of the line?"
]

# Linear Function Word Problems (Multiple Choice) (May need image-gen)
# ChatGPT GPT-4 Success Rate For Khan Academy Questions 4/6
# GPT-4 Success Rate For Generated Questions 
# Caveat * GPT Generated the correct answer, but selected the wrong answer choice. May need to use CoT prompting
sample_questions_5 = [
    "A utility service company has a fleet of cars and trucks. The gas tank of each car has a volume of 16 gallons, and the gas tank of each truck has a volume of 30 gallons. If it takes 2,000 gallons of gas to fill the empty gas tanks of the entire fleet, which equation shows the possible number of cars, c, and number of trucks, t, in the fleet? Choose 1 answer: (A) 16t + 30c = 2,000, (B) 30t + 16c = 2,000, (C) t + 30c = 2,000, (D) t + 16c = 2,000"
    "When Lúcia plays golf at the Rolling Hills golf course, she loses about 12 balls on average. When she plays golf at the Meandering Meadows golf course, she loses about 8 balls on average. If Lúcia lost approximately 100 balls after playing at Rolling Hills r times and at Meandering Meadows m times, which of the following equations best represents the relationship between r and m? Choose 1 answer: (A) 8r + 12m = 100, (B) 12r + 8m = 100, (C) 12r - 8m = 100, (D) 8r - 12m = 100",
    "A farmer earns on average $5,000 per acre of raspberries grown and $11,000 per acre of strawberries grown. Which of the following equations shows the relationship between the number of acres of raspberries, r, and the number of acres of strawberries, s, that the farmer can harvest to earn a total of $50,000 from these two crops? Choose 1 answer: (A) 5,000r + 11,000s = 50,000, (B) 11,000r + 5,000s = 50,000, (C) (r/5,000) + (s/11,000) = 50,000, (D) (s/5,000) + (r/11,000) = 50,000",
    "A clothier who only makes shirts and pants can make a shirt in 4 hours and a pair of pants in 6 hours. Which of the following equations shows the number of shirts, s, and the number of pants, p, that the clothier can make in a 40 hour work week? Choose 1 answer: (A) 6s + 4p = 40, (B) 4s + 6p = 40, (C) (s/4) + (p/6) = 40, (D) (s/6) + (p/4) = 40",
    "A farmer has a rectangular plot with a length of l meters and a width of w meters. The total perimeter of the plot is 700 meters. Which equation represents the perimeter of the plot in terms of length and width? Choose 1 answer: (A) 2l + 2w = 700, (B) 2l + 2w = 350, (C) l + w = 700, (D) lw = 700"
]

# System of linear inequalities word problems (Multiple-Choice)
# ChatGPT GPT-4 Success Rate For Khan Academy Questions 5/5
# GPT-4 Success Rate For Generated Questions 
# Caveat * GPT Generated the correct answer, but selected the wrong answer choice. May need to use CoT prompting
sample_questions_6 = [
    "In order to bring his business to the next level, Christov wants to gain at least 2,000 followers on a popular social media platform. From his own personal account, he knows that each original post gains him approximately 3 new followers and every 5 reposts gains about 1. Which of the following inequalities represents the numbers of posts, P, and reposts, R, Christov needs to reach his goal of gaining at least 2,000 followers? Choose 1 answer: (A) 3P + 0.2R ≥ 2,000, (B) 3P + 5R ≤ 2,000, (C) 1P + 5R ≥ 2,000, (D) 0.2P + 5R ≤ 2,000",
    "Joe is buying apples and persimmons at the grocery store. Each apple costs $0.99 and each persimmon costs $0.79. If Joe has $10, which of the following inequalities describes x, the number of apples, and y, the number of persimmons, that he can buy? Choose 1 answer: (A) 0.79x + 0.99y ≥ 10, (B) 0.99x + 0.79y ≥ 10, (C) 0.79x + 0.99y ≤ 10, (D) 0.99x + 0.79y ≤ 10",
    "A trivia contest asks both multiple choice and free response questions. Contestants receive 3 points for each correct multiple choice question and 5 points for each correct free response question, and they must score more than 60 points to advance to the next round. If Eva advanced to the next round of the contest, which of the following inequalities describes x, the number of multiple choice questions, and y, the number of free response questions, that she answered correctly? Choose 1 answer: (A) 3x + 5y > 60, (B) 3x + 5y ≥ 60, (C) 3x + 5y < 60, (D) 3x + 5y ≤ 60",
    "Dante commutes to work 4 mornings a week. For his commute each morning, he walks for 10 minutes, waits and rides the bus for x minutes, and waits and rides the train for y minutes. If Dante spends at least 3.5 hours on his morning commute each week, which of the following inequalities best describes Dante's weekly morning commute? Choose 1 answer: (A) x + y + 10 ≥ 3.5(60), (B) x + y + 10 ≥ 3.5(60)(4), (C) 4(x + y) + 10 ≥ 3.5(60), (D) 4(x + y + 10) ≥ 3.5(60)",
    "Vanessa has a $900 travel and lodging budget for her vacation. She found round-trip plane tickets for x dollars total, a hotel for y dollars per night, and free shuttle service between the airport and the hotel. If she plans to stay at the hotel for 5 nights, and she spends less than what she budgeted on travel and lodging, which of the following inequalities best describes the scenario? Choose 1 answer: (A) x + 5y > 900, (B) 5x + y > 900, (C) x + 5y < 900, (D) 5x + y < 900"
]

# Solving Systems of Linear Equations
# ChatGPT GPT-4 Success Rate For Khan Academy Questions 5/5
# GPT-4 Success Rate For Generated Questions 
# Caveat * GPT Generated the correct answer, but selected the wrong answer choice. May need to use CoT prompting
sample_questions_7 = [
    "The system of equations above has solution (x,y). Equations: x + 3y = 2 and 4x - 3y = 23. What is the value of x? Choose 1 answer: (A) -1, (B) 5, (C) 7, (D) 25.",
    "If (x,y) satisfies the system of equations 3x - 4y = 10 and 2x - 4y = 6, what is the value of y? Choose 1 answer: (A) 1/10, (B) 1/2, (C) 16/5, (D) 4.",
    "Which ordered pair (x,y) satisfies the system of equations 5x - y = 3 and -5x + 2y = 4? Choose 1 answer: (A) (2/5, -1), (B) (4/5, 1), (C) (1, 2), (D) (2, 7).",
    "What is the solution (x,y) to the system of equations x + y = 3 and x - 3y = -9?",
    "The system of equations 3x + y = 3 and 7x - y = 2 has solution (x,y). What is the value of x?"
]

# System of linear equations word problems
# ChatGPT GPT-4 Success Rate For Khan Academy Questions 5/5
# GPT-4 Success Rate For Generated Questions 
# Caveat * GPT Generated the correct answer, but selected the wrong answer choice. May need to use CoT prompting
sample_questions_8 = [
    "Ricardo has two types of assignments for his class. The number of mini assignments, m, he has is 1 fewer than twice the number of long assignments, l, he has. If he has 46 assignments in total, which of the following systems of equations can be used to correctly solve for m and l? Choose 1 answer: (A) m=2l-1, m+l=46, (B) m=2l-1, m=l+46, (C) l=2m-1, m+l=46, (D) l=2m-1, m=l+46.",
    "A piece of glass with an initial temperature of 99°C is cooled at a rate of 3.5°C per minute. At the same time, a piece of copper with an initial temperature of 0°C is heated at 2.5°C per minute. Which of the following systems of equations can be used to solve for the temperature, T, in degrees Celsius, and the time, m, in minutes, when the glass and copper reach the same temperatures? Choose 1 answer: (A) T = 99 + 3.5m, T = 2.5m, (B) T = 99 - 3.5m, T = 2.5m, (C) T = 99 + 2.5m, T = 3.5m, (D) T = 99 - 2.5m, T = 3.5m.",
    "Ethan sells e candy bars for $2.50 apiece and Chloe sells c candy bars for $2.00 apiece to raise money for a school trip. Ethan sold 15 fewer candy bars than Chloe, but he also got a $6.00 donation. If Chloe and Ethan raised the same amount of money, which of the following systems could be used to find how many candy bars each sold? Choose 1 answer: (A) 2c = 2.5e + 6, c = e - 15, (B) 2c = 2.5e + 6, e = c - 15, (C) 2c + 6 = 2.5e, c = e - 15, (D) 2c + 6 = 2.5e, e = c - 15",
    f"""Liam deposits l dollars into an account that earns 0.9% in simple interest each year. Grace deposits g dollars into an account that earns 1.1\\% in simple interest each year. Both Liam and Grace let their money earn interest for one year and make no further deposits. If Liam's initial deposit was $800 more than Grace's, and if both Liam and Grace earn the same amount of interest after one year, which of the following systems of equations could be used to find their initial deposits? Choose 1 answer: (A) 0.009l + 800 = 0.011g, l = g, (B) 0.009l = 0.011g + 800, l = g, (C) 0.009l = 0.011g, l + 800 = g, (D) 0.009l = 0.011g, l = g + 800""",
    "Eva maintained an average speed of 35 mph for the first m hours of her road trip. For the next n hours of the trip, she drove at an average speed of 60 mph. Eva drove a total of 225 miles in 4.5 hours. Which of the following systems of equations could be used to find how many miles Eva drove in the first m hours of the trip? Choose 1 answer: (A) m + n = 225, 35m + 60n = 4.5; (B) m + n = 4.5, 35m + 60n = 225; (C) m + n = 225, 60m + 35n = 4.5; (D) m + n = 4.5, 60m + 35n = 225"
]

### HEART OF ALGEBRA END

#
#
#
#

### Passport to Advanced Mathmatics Start


# Solving Quadratic Equations
# ChatGPT GPT-4 Success Rate For Khan Academy Questions 5/5
# GPT-4 Success Rate For Generated Questions 
# Caveat * GPT Generated the correct answer, but selected the wrong answer choice. May need to use CoT prompting
sample_questions_9 = [
    "72=2x^2; What are the solutions to the equation above? Choose 1 answer: (A) x=6 only, (B) x=−6 and x=6, (C) x=−2+√2 and x=−2−√2, (D) x=−2+√2 and x=−2−√2",
    "3n^2 = 27; What are the solutions to the equation above? Choose 1 answer: (A) n=√3, (B) n=3, (C) n=−√3 and n=√3, (D) n=−3 and n=3",
    "100−121k^2 = 0; What are the solutions to the equation above? Choose 1 answer: (A) k=100/121, (B) k=−100/121 and k=100/121, (C) k=10/11, (D) k=−10/11 and k=10/11",
    "−81x^2 =−11; What are the solutions to the equation above? Choose 1 answer: (A) x=√(11/81), (B) x=−√(11/81) and x=√(11/81), (C) x=−√(11/9) and x=√(11/9), (D) x=√(11/9)",
    "0=32−50x^2; What are the solutions to the equation above? Choose 1 answer: (A) x=4/5, (B) x=−4/5 and x=4/5, (C) x=16/25, (D) x=−16/25 and x=16/25"
]

# Interpreting nonlinear expressions (May need to draw a visual for given equations)
# ChatGPT GPT-4 Success Rate For Khan Academy Questions 5/5 
# GPT-4 Success Rate For Generated Questions 
# Caveat * GPT Generated the correct answer, but selected the wrong answer choice. May need to use CoT prompting
sample_questions_10 = [
    "h=0.3+5.5t−4.9t^2; What was the height of the football in meters at the moment of the kick?",
    "F(t)=1,500(1.045)^t; The future value in dollars, F(t), of an investment after t years is given by the function defined above. What is the initial value of the investment in dollars?",
    "h(x)=0.000371(x^2 −1,280x)+152; The Golden Gate Bridge is a suspension bridge that consists of two cables hung from two towers of equal height that are 1,280 meters apart. The function above models h, the height of each cable above the ground in meters, as it relates to x, the cable's horizontal distance from the left tower in meters. What is the height of the towers in meters? Choose 1 answer: (A) 640, (B) 152, (C) 0.000371, (D) 1280",
    "c=0.4⋅0.9^m; The zebra mussel, Dreissena polymorpha, filters particulate organic carbon (POC) from water as part of its feeding pattern. The concentration, c, of POC (in milligrams per liter) remaining in a particular bay m months after the introduction of a population of zebra mussels can be estimated using the following equation. According to this estimate, how many milligrams per liter of POC were in the bay when the zebra mussels were first introduced?",
    "A(x)=-1/4(x−25)^2 +625; The area, A(x), of a rectangular enclosure that can be made from a limited amount of fencing is shown above, where x is the length of one of the sides of the enclosure, measured in feet. What is the maximum area that can be enclosed in square feet?"
]

# Quadratic and Exponential Word Problems
# ChatGPT GPT-4 Success Rate For Khan Academy Questions 5/5
# GPT-4 Success Rate For Generated Questions 
# Caveat * GPT Generated the correct answer, but selected the wrong answer choice. May need to use CoT prompting
sample_questions_11 = [
    "T(t)=22+53(0.74)^t; Poultry should be cooked to a temperature of 75°C. A chicken is removed from the oven and left to rest in a room that is at a constant temperature of 22°C. The temperature of the chicken t hours after it is removed from the oven is given by the exponential function above. What is the approximate temperature of the chicken after 2 hours? Choose 1 answer: (A) 22°C, (B) 51°C, (C) 74°C, (D) 75°C",
    "h(t)=56−4.9t^2; The function above models h, the height of a flower pot in meters, t seconds after it falls from a fourth floor balcony. What is the height of the flower pot, in meters, 3 seconds after it falls? Choose 1 answer: (A) 51.1, (B) 44.1, (C) 36.4, (D) 11.9",
    "f(x)=0.145x^2; The function above models f, the kinetic energy, in joules, of a baseball traveling at a speed of x meters per second. Based on the function, what is the kinetic energy, in joules, of a baseball traveling at a speed of 40 meters per second? Choose 1 answer: (A) 5.8, (B) 58, (C) 232, (D) 2,320",
    "P(t)=1,800(1.004)^t; The function above models P, the amount of money, in dollars, in Yara's savings account t years after she opened the account with an initial deposit of $1,800. How much money is in Yara's account 5 years after her initial deposit if she makes no deposits or withdraws in that time? Choose 1 answer: (A) $1,836.29, (B) $1,873.31, (C) $2,189.98, (D) $9,036",
    "h(t)=-16.1t^2 +100t; The equation above models h, the height of a firework shell in feet, t seconds after launch. What is the height, in feet, of the firework shell 2 seconds after launch? Choose 1 answer: (A) 135.6, (B) 167.8, (C) 232.2, (D) 264.4"
]

# Manipulating Quadratic and Exponential Expressions
# ChatGPT GPT-4 Success Rate For Khan Academy Questions 3/5
# GPT-4 Success Rate For Generated Questions 
# For question 3/4, gpt got completely wrong
sample_questions_12 = [
    "y=2x^2 -5x + 7; If the equation is graphed in the xy-plane, which of the following characteristics of the graph is displayed as a constant or coefficient in the equation? Choose 1 answer: (A) x-intercept(s), (B) y-intercept, (C) x-coordinate of the vertex, (D) y-coordinate of the vertex",
    "y=(x−1)(x+5); If the equation is graphed in the xy-plane, which of the following characteristics of the graph is displayed as a constant in the equation? Choose 1 answer: (A) x-coordinate of the vertex, (B) x-intercept(s), (C) Maximum y-value, (D) y-intercept",
    "y=−(x−1)^2 +3; If the equation is graphed in the xy-plane, which of the following characteristics of the graph is displayed as a constant or coefficient in the equation? Choose 1 answer: (A) y-intercept, (B) x-intercept(s), (C) Minimum y-value, (D) x-coordinate of the line of symmetry",
    "If y = (x + 2)(x + 8) is graphed in the xy-plane, which of the following characteristics of the graph is displayed as a constant in the equation? Choose 1 answer: (A) x-intercept(s), (B) y-intercept, (C) x-coordinate of the vertex, (D) Minimum y-value",
    "If y = - (1/2) x^2 - 9 is graphed in the xy-plane, which of the following characteristics of the graph are displayed as a constant or coefficient in the equation? I. x-intercept(s) II. y-intercept III. y-coordinate of the vertex Choose 1 answer: (A) II only, (B) III only, (C) I and II only, (D) II and III only"
]

# Radicals and rational exponents
# ChatGPT GPT-4 Success Rate For Khan Academy Questions 5/5
# GPT-4 Success Rate For Generated Questions 
sample_questions_13 = [
    "b^3 * (b^4)^2 = b^x, what is the value of x? Choose 1 answer: (A) 9, (B) 11, (C) 18, (D) 19",
    "( (1/2)^(-2) +3^0 ) What is the value of the expression above? Choose 1 answer: (A) 3/4, (B) 5/4, (C) 4, (D) 5",
    "((a^3)^3 * a^(-9)) Which of the following expressions is equivalent to the expression above for all a ≠ 0? Choose 1 answer: (A) 0, (B) 1, (C) a^3, (D) a^18",
    "(2y^2 z^10)^5 Which of the following expressions is equivalent to the expression above? Choose 1 answer: (A) 10y^32 z^100,000, (B) 10y^10 z^50, (C) 32y^32 z^100,000, (D) 32y^10 z^50",
    "-(5c)^0 + 7^1 - 2^2 If c ≠ 0, what is the value of the expression above?"
]

# Radicals and rational equations
# ChatGPT GPT-4 Success Rate For Khan Academy Questions 5/5
# GPT-4 Success Rate For Generated Questions 
sample_questions_14 = [
    "√x = √(3x) What is the solution to the equation above?",
    "n + 2 = √(a - n) In the equation above, a is a constant. If n = 1 is a solution to the equation, what is the value of a?",
    "7x = 13√x What is the greatest value of x that is a solution to the above equation? Choose 1 answer: (A) 0, (B) 49/169, (C) 169/49, (D) 7/13",
    "w = √(108w) What is the sum of all solutions to the above equation?",
    "√s + 7 = 6 + 4√s What is the solution to the equation above?"
]

# Operations with rational expressions
# ChatGPT GPT-4 Success Rate For Khan Academy Questions 3/5
# GPT-4 Success Rate For Generated Questions 
# Question 3 + Q5 Failed - However, asking gpt-4 to generate code and subsequently running the code causes it to answer questions correctly - used sympy library
sample_questions_15 = [
    "3x/(2b) - 5x/(6b) Which of the following expressions is equivalent to the expression above for b ≠ 0 ? Choose 1 answer: (A) −x/(6b), (B) −x/(3b), (C) x/(2b), (D) 2x/(3b)",
    "3/(14y) + y/14 Which expression is equivalent to the above sum for y ≠ 0 ?",
    '(8v/28w+21 − 3v+10/4w+3, "Which expression is equivalent to the above difference? Choose 1 answer:", "A: (8v^2 − 21v + 10) / (24w + 18)", "B: (-13v + 10) / (24w + 18)", "C: (53v + 10) / (28w + 21)", "D: (-13v − 70) / (28w + 21)")',
    "x−5/7 + 5−x/4 is simplified to which expression for all x ≠ 5? Choose 1 answer: (A) 11/(x−5), (B) 11/(5−x), (C) 3/(x−5), (D) 3/(5−x)",
    "x^2/(x−2) + 4/(2−x) is simplified to which expression for all x ≠ 2? Choose 1 answer: (A) x+2, (B) x−2, (C) (x^2−4)/(2−x), (D) (x^2+4)/(x−2)"
]

# Operations with Polynomials
# ChatGPT GPT-4 Success Rate For Khan Academy Questions 6/6
# GPT-4 Success Rate For Generated Questions 
sample_questions_16 = [
    "7n - (4n - 3), which of the following is equivalent to the expression above? Choose 1 answer: (A) 3n + 3, (B) 3n - 3, (C) 11n + 3, (D) 11n - 3",
    "(x−6)(x−1), which of the following is equivalent to the expression above? Choose 1 answer: (A) x^2 - 7x + 6, (B) x^2 + 5x - 7, (C) 2x^2 - 7x + 6, (D) 2x^2 - 7x - 7",
    "(x+4)(x−3), which of the following is equivalent to the expression above? Choose 1 answer: (A) x^2 + x + 1, (B) x^2 + x −12, (C) 2x^2 + x −12, (D) 2x^2 + 7x + 1",
    "(x−4)(x−8), which of the following is equivalent to the expression above? Choose 1 answer: (A) x^2 − 12x + 32, (B) 2x^2 + 4x + 32, (C) x^2 + 4x − 12, (D) 2x^2 − 12x + 32",
    "(x−5)(x+7), which of the following is equivalent to the expression above? Choose 1 answer: (A) x^2 + 2x − 35, (B) x^2 + 2x + 2, (C) x^2 − 35, (D) 2x^2 − 12x − 35"
]

# Polynomial Factors and Graphs
# ChatGPT GPT-4 Success Rate For Khan Academy Questions 5/5
# GPT-4 Success Rate For Generated Questions 
sample_questions_17 = [
    "(x−7)(x+5)(2x−3)=0, Given the polynomial above, what are its zeros? Choose 1 answer: (A) {−7,5,−3}, (B) {7,−5,3}, (C) {−7,5,−3/2}, (D) {7,−5,3/2}",
    "A polynomial function f is defined as f(x) = 3(5x+3)(x+2)(7x-1). Which of the following is a zero of function f? Choose 1 answer: (A) -3, (B) -2, (C) 2, (D) 3"
    "A polynomial function M is defined as M(x) = (2x - 3)(x^2 + 3x + 10). If M(a) = 0 for some real number a, then what is the value of a?",
    "The polynomial function h(t) is defined as h(t) = (t-8)^1(t-4)^2(t-2)^3(t-1)^4. How many distinct zeros does h(t) have?",
    "f(x) = x(6 - x)(x + 1)(x + 2). Which of the following is a zero of the function above? Choose 1 answer: (A) 1, (B) 2, (C) 6, (D) 12"
]

# Nonlinear equation graphs
# ChatGPT GPT-4 Success Rate For Khan Academy Questions 4/5
# GPT-4 Success Rate For Generated Questions 
# Question 5 wrong -> Maybe use codegen and sympy instead?
sample_questions_18 = [
    "The equation y = 2x^2 - 7x + 1 is graphed in the xy-plane. What is the y-intercept of the graph? Choose 1 answer: (A) -1, (B) 1, (C) 2, (D) 7",
    "y = (x - 3)(x + 9). The equation above is graphed in the xy-plane. Which of the following are x-intercepts of the graph? Choose 1 answer: (A) -3 and -9, (B) -3 and 9, (C) 3 and -9, (D) 3 and 9",
    "f(x) = (x - 3)^2 - 4. The graph of the function above is a parabola. What are the coordinates of the vertex of the parabola? Choose 1 answer: (A) (-3, -4), (B) (-3, 4), (C) (3, -4), (D) (3, 4)",
    "If the equation y = 2(1.5)^x is graphed in the xy-plane, what are the coordinates of its y-intercept? Choose 1 answer: (A) (0, 0), (B) (0, 1.5), (C) (0, 2), (D) (0, 3)",
    "The function f(x) = (2x - 1)(3x + 7) is graphed in the xy-plane. Which of the following are the coordinates of an x-intercept of the graph? Choose 1 answer: (A) (-3/7, 0), (B) (1/2, 0), (C) (2, 0), (D) (7/3, 0)"
]

# Linear and Quadratic Systems (Need to use matplotlib to generate a graph)
# ChatGPT GPT-4 Success Rate For Khan Academy Questions 5/5 - USED CODEGEN FOR ALL 5
# GPT-4 Success Rate For Generated Questions 
sample_questions_19 = [
    "A quadratic equation is given by y = (1/2)x^2 - 6. Which of the following equations could be paired with the graphed equation to create a system of equations whose solution set is comprised of the points (2, -4) and (-4, 2)?",
    "A quadratic equation is given by y = (-1/2)x^2 + 2x + 1. Which of the following linear equations combines with the graphed equation to create a system of equations whose solutions are the points (3, 5/2) and (-2, -5)? Choose 1 answer: (A) 3x + 2y = -4, (B) x + 2y = 8, (C) 2x - y = 1, (D) 3x - 2y = 4",
    "Consider the following system of equations: y = -3/5x + 3, y = (x - 5)(x + k) where k is a constant. If the solutions to this system of equations are the points (0, 3) and (5, 0), what is the value of k? Choose 1 answer: (A) 3, (B) -3/5, (C) 3/5, (D) -3",
    "A linear equation is given by y = 2x - 3. Which of the following equations combines with the graphed equation to create a system of equations whose solution set is comprised of the points (-1, -5) and (3, 3)? Choose 1 answer: (A) y = x^2 - 6, (B) y = 6 - x^2, (C) y = 4 - x^2, (D) y = x^2 - 4"
    "A quadratic equation is given by y^2 = 6 - x. Which of the following equations could be paired with the graphed equation to create a system of equations whose solution set is comprised of the points (2, -2) and (-3, 3)? Choose 1 answer: (A) y = x + 6, (B) y = x - 6, (C) y = x, (D) y = -x"
]

# Structure in expressions (Need to use matplotlib to generate a graph)
# ChatGPT GPT-4 Success Rate For Khan Academy Questions 5/5 - USED CODEGEN FOR ALL 5
# GPT-4 Success Rate For Generated Questions 
sample_questions_20 = [
    "Which of the following is equivalent to the expression x^2 + 11x + 24? A) (x+2)(x+12), B) (x+3)(x+8), C) (x+4)(x+6), D) (x+5)(x+6)",
    "Which of the following is equivalent to the expression x^2 - 5x - 14? A) (x-14)(x+1), B) (x-7)(x+2), C) (x-2)(x+7), D) (x-1)(x+14)",
    "Which of the following is equivalent to the expression x^2 + 3x - 10? A) (x-2)(x-5), B) (x-2)(x+5), C) (x+2)(x-5), D) (x+2)(x+5)",
    "Which of the following is equivalent to the expression x^2 - 8x + 15? A) (x-3)(x-5), B) (x-3)(x+5), C) (x+3)(x-5), D) (x+3)(x+5)",
    "Which of the following is equivalent to the expression x^2 + 30x + 200? A) (x+4)(x+50), B) (x+5)(x+25), C) (x+8)(x+25), D) (x+10)(x+20)"
]

# Isolating Quantities
# ChatGPT GPT-4 Success Rate For Khan Academy Questions 5/5 - USED CODEGEN FOR ALL 5 (Advanced Data analysis) - however it did not gen code
# GPT-4 Success Rate For Generated Questions 
sample_questions_21 = [
    "Which of the following equations correctly expresses g in terms of f and h? A) g = f / (12h+15), B) g = f / (12h-15), C) g = (f-15) / 12h, D) g = f / 27h",
    "Which of the following equations correctly expresses c in terms of j and m? A) c = (m/j) * 78, B) c = (78*j)/m, C) c = (m/j) * 78, D) c = (78*m)/j",
    "Which of the following equations correctly expresses n in terms of l and m? A) n = (l-125)/(50m), B) n = l/50 - 125/m, C) n = l/50 - 5/(2m), D) n = (5/(2m)) - (l/50)"
    "Which of the following correctly shows the trapezoid's height in terms of its area and two bases? A) h = (A / 2) * (b1+b2), B) h = (a) * ((b1+b2) / 2), C) h = A * (2 * (b1+b2)), D) h = (2A) / (b1+b2)",
    "Which of the following is the correct equation for the distance in terms of the angle and the wavelength? A) d = λ / (2sin(θ)), B) d = 2λ / sin(θ), C) d = λ / (2sin(θ)), D) d = 2λ / sin(θ)"
]

# Function notation
# ChatGPT GPT-4 Success Rate For Khan Academy Questions 5/5 - USED CODEGEN FOR ALL 5 (Advanced Data analysis) 
# GPT-4 Success Rate For Generated Questions 
# Question 5 wrong -> Maybe use codegen and sympy instead?
sample_questions_22 = [
    "If f(x)=3x-1 and g(x)=x^2+1, what is the value of g(f(3))? A) 8, B) 10, C) 29, D) 65",
    "The functions g and h are defined as g(x) = 3x - 7 and h(x) = 2 - g(x). What is the value of h(1)? A) 6, B) 1, C) -2, D) -6",
    "A function f satisfies f(1)=3 and f(3)=7. A function g satisfies g(3)=9 and g(7)=1. What is the value of f(g(7))? A) 1, B) 3, C) 7, D) 9",
    "The functions f and g are defined as f(x) = 1/(x-1) and g(x) = 5x+8. What is the value of f(g(-1))?",
    "If h(x) = x^3 - 4x + 3, what is the value of h(h(2))?"
]

category = passport_to_advanced_math_categories[-1]

system_msg = f""" You are an AI agent designed to generate SAT level difficulty math questions for a given category of math. "
Ensure that these questions cover the breadth of the passage and that they can help augment SAT test taker's reading comprehension ability.
Ensure that the questions vary in difficulty, from 1 to 5, where 1 is easy and 5 is very difficult, and help challenge prospective test takers.
I want a format similar to {newString} and make sure it has the right amount of questions as specified. You will either generate questions with multiple
choice
"""

user_content1 = f"""Given the category: {category}, generate a suite of 11 SAT Math questions as well as 4-5 answer choices for each question only if the format specified is multiple-choice. Return the questions,
their answer choices, the correct answer choice, a brief rationale for the correct answer, and an estimated difficulty score for each question. 
The following represents a list of sample questions that you can use to generate new questions from: {sample_questions_22}. Make the questions harder than the list of sample questions.
"""

# user_content2 = f"""
# Given the category: {category}, generate a suite of 11 SAT Math questions as well as 4-5 answer choices for each question. Return the questions,
# their answer choices, the correct answer choice, a brief rationale for the correct answer, and an estimated difficulty score for each question. 
# The following represents a list of sample questions that you can use to generate new questions from: {sample_questions_1}
# """

# with open("questionAns.json", "r") as f:
#     dataDict = json.load(f)
#     assistant_content_1 = json.dumps(dataDict, indent=4)

def generate_math_question():
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content1},
        # {"role": "assistant", "content": assistant_content_1},
        # {"role": "user", "content": user_content2}
        ]
    )

    response_content = completion.choices[0]['message']['content']
    parsed_content = json.loads(response_content)  # Parsing string to dictionary

    with open("answers/sample_answers_22.json", "w") as f:
        json.dump(parsed_content, f, indent=4)  # Dumping dictionary with formatting

    return json.dumps(parsed_content, indent=4)

question = generate_math_question()

