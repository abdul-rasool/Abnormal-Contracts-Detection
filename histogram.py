import matplotlib.pyplot as plt
import psycopg2

# An "interface" to matplotlib.axes.Axes.hist() method
##n, bins, patches = plt.hist(x=d, bins='auto', color='#0504aa',
##                            alpha=0.7, rwidth=0.85)


##x = [1,1,2,3,3,5,7,8,9,10,
##     10,11,11,13,13,15,16,17,18,18,
##     18,19,20,21,21,23,24,24,25,25,
##     25,25,26,26,26,27,27,27,27,27,
##     29,30,30,31,33,34,34,34,35,36,
##     36,37,37,38,38,39,40,41,41,42,
##     43,44,45,45,46,47,48,48,49,50,
##     51,52,53,54,55,55,56,57,58,60,
##     61,63,64,65,66,68,70,71,72,74,
##     75,77,81,83,84,87,89,90,90,91
##     ]

conn = psycopg2.connect("host=localhost dbname=postgres user=postgres password=asd")
cur3 = conn.cursor()
cur3.execute("select length(code) from ponzi_D111_new2 where label='1'")
rows2 = cur3.fetchall()
x = []

for i in rows2:
    x.append(int(i[0]))



plt.grid(axis='y', alpha=0.75)
plt.xlabel('Bytecode length')
plt.ylabel('Number of samples')
plt.title('Distribution of Ponzi contracts')
plt.hist(x, bins='auto',color='#0504aa', alpha=0.7, rwidth=0.85 )
plt.show()


##plt.text(23, 45, r'$\mu=15, b=3$')
##maxfreq = n.max()
### Set a clean upper y-axis limit.
##plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
