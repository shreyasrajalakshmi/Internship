#take the user input temperature in the format (e.g., 50F, 100C, etc.)
temp = input("Input the temperature you like to convert? (e.g., 50F, 100C etc.) :")

#extracting only the numerical part of the temperature and convert it to an integer
degree = int(temp[:-1])

#extracting the convention part of the temperature input (either 'C' or 'F')
iconvention = temp[-1]

#check if the input convention is in uppercase 'C' (Celsius)
if iconvention.upper() == "C":
    #Celsius temperature to Fahrenheit
    result = int(round((9 * degree) / 5 + 32))
    o_convention = "Fahrenheit"  #set the output convention as Fahrenheit

#check if the input convention is in uppercase 'F' (Fahrenheit)
elif iconvention.upper() == "F":
    #Fahrenheit temperature to Celsius
    result = int(round((degree - 32) * 5 / 9))
    o_convention = "Celsius"  #set the output convention as Celsius

else:
    #wrong input
    print("Input proper convention.")
    quit()

#display the converted temperature in the specified output convention
print("The temperature in", o_convention, "is", result, "degrees.") 