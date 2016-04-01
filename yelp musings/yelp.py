import json
import pandas as pd
from itertools import chain
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

#I was working on the Yelp academic dataset, see here http://www.yelp.com/dataset_challenge. 
#And I settled on a problem of finding ways to identify businesses that are kid-friendly 
#just from the content of their review. The problem morphed into analysing the differences 
#between reviews that contain kid-related keywords and those that do not.

keyword=['kids','child','children','baby','teenager','son','daughter','kid', 'family','mom','moms','toddler','toddlers']

with open("yelp_academic_dataset_review.json", "r") as f: #open json file and use data.
    data=f.readlines()
yelp= pd.DataFrame(json.loads(i) for i in data)
nrows = len(yelp)
print "Yelp dataset has ", nrows," rows and the following columns fields:"
print yelp.columns
print "-----------------------------------------------------"
print "The first five rows of the dataset: "
print  yelp.head()
print "-----------------------------------------------------"

#classify a review based on if it contains any of the above keywords. I have chosen to ignore the star rating for now.
def IsReviewKidFriendly(a):
    return set(a.lower().split()).isdisjoint(set(keyword))
yelp.loc[:,'kids']=~yelp['text'].apply(IsReviewKidFriendly) 


#restrict yelp data to the following columns.
results =yelp[['stars','kids']]
#Add columns for funny, useful and cool votes for review
results.loc[:,'funny'] = [yelp['votes'].iloc[s]['funny'] for s in range(nrows)]
results.loc[:,'useful'] = [yelp['votes'].iloc[s]['useful'] for s in range(nrows)]
results.loc[:,'cool'] = [yelp['votes'].iloc[s]['cool'] for s in range(nrows)]
results.loc[:,'count']=[1]*nrows
results.loc[:,'business_id'] = yelp['business_id']
del yelp #delete yelp data.





# Generate a pivot table using kids and stars and as the new index.
jovely=results.pivot_table(index=['kids','stars'],aggfunc=sum)
print "Pivot table indexed by kids and stars column:"
print jovely.head()
print "--------------------------------------------------"
#Generate a normalised bar plot of review stars grouped by whether the review is labeled kids-related or not
N = 5 
sm=jovely.query('kids==True')['count'].sum()  # number of kids-related reviews
normt = (jovely.query('kids==True')['count']/sm).reset_index()
kid = normt['count'].values #normalised frequency counts of the star ratings 
ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(ind, kid, width, color='r',alpha=0.7)


#Plot the bar plot for the second group
smf=jovely.query('kids==False')['count'].sum()
normf = (jovely.query('kids==False')['count']/smf).reset_index()
notkid = normf['count'].values
rects2 = ax.bar(ind+width, notkid, width, color='y',alpha=0.5)


# add some text for labels, title and axes ticks
ax.set_ylabel('Probability')
ax.set_title('Distribution of Business Star Ratings.')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('1', '2', '3', '4', '5') )
ax.legend( (rects1[0], rects2[0]), ('Kid-related reviews', 'Non Kid-related reviews'),loc = 'upper left' )
plt.show()



#Generate a histogram of vote counts for the kid-related reviews and non-kid-related reviews
results.loc[:,'tots'] = results[['funny','useful','cool']].sum(axis=1) #record a vote count for each review
fig, ax = plt.subplots()
plt.hist(results.query('kids==True')['tots'].values,bottom=0.01, bins=50,color='b',alpha=0.8,label='kid-friendly reviews')
plt.hist(results.query('kids==False')['tots'].values,bottom=0.01,bins=50, color='r', alpha=0.5,label='Non kid-friendly reviews')
plt.xlabel('Aggregate vote counts')
plt.ylabel('Frequency')
ax.set_yscale('log')
plt.title("Vote Counts Histogram")
plt.legend()

#Get a distribution of the business categories for the two review groups


#In order to extract business categories, I have to cross-reference business ids of 
#the reviews with another yelp dataset that contains the business categories.
with open("yelp_academic_dataset_business.json", "r") as f:
    data=f.readlines()
business_dat= pd.DataFrame(json.loads(i) for i in data)
business_dat.set_index(keys='business_id',inplace=True) #reindex data using the business-id as the new keys


good_buis=results[results['kids']==True]['business_id'].unique() #business ids of reviews that are kid-related
good_categories=[business_dat.loc[s]['categories'] for s in good_buis] #retrieve the categories for businesses labeled as kid-friendly
good_categories=list(chain(*good_categories)) #flatten the two level list.
good_categories_freq = Counter(good_categories) #get the frequency list of the categories
print "The fifteen most popular categories of business labeled as kid friendly, "
print good_categories_freq.most_common(15)
print "----------------------------------------------------------"



#Let's try to make a bubble chart of the 15 most popular business categories.
n=15
txt=np.array([good_categories_freq.most_common(i)[-1][0] for i in range(1,n+1)])
y=np.array([good_categories_freq.most_common(i)[-1][1] for i in range(1,n+1)])
x=np.arange(n)
np.random.shuffle(x)
cm = plt.cm.get_cmap('jet')
colors = np.random.rand(n)
fig, ax = plt.subplots()

#plot the text label first
for j,i in enumerate(x):
    ax.text(j,y[i],txt[i][0:11],size=6,horizontalalignment='center',stretch="ultra-condensed")
plt.scatter(np.arange(n),y[x],s=y[x],c=colors,linewidth=0,alpha=0.5)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.axes.set_xlabel('x')
ax.axes.set_ylabel('y')
ax.axes.set_title('Most common business categories with kids-related reviews');
ax.set_ylim([0,15000])
plt.show()




