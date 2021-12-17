import musicalgestures

# mg = musicalgestures.MgObject('adl40.avi', color=False) # starttime=1, endtime=2)
# #mg.show() # view the result
# mg.motion()

for i in range(1,31):
    two_digit_i = (f"{i:02}")
    print(f'fall{two_digit_i}.avi')

    mg = musicalgestures.MgObject(f'fall{two_digit_i}.avi', color=False) # starttime=1, endtime=2)
    mg.motion()


for i in range(1,41):
    two_digit_i = (f"{i:02}")
    print(f'adl{two_digit_i}.avi')

    mg = musicalgestures.MgObject(f'fall{two_digit_i}.avi', color=False) # starttime=1, endtime=2)
    mg.motion()
