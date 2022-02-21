import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Solution:
    def __init__(self)->None:
           url = 'https://raw.githubusercontent.com/sithu/cmpe255-spring21/main/lab1/data/chipotle.tsv'
           df = pd.read_csv(url,'\t')
           file = 'data/chipotle.tsv'
           self.chipo = pd.read_csv(url,'\t')
           
    
    def top_x(self, count) -> None:
         topx =self.chipo.head(count)
         print(topx.to_markdown())
    
    
    def count(self) -> int:
        if len(self.chipo)>0:
            return len(self.chipo)
        return -1
    
    def info(self) -> None:
        print(self.chipo.info())
        pass
    
    def num_column(self) -> int:
        # TODO return the number of columns in the dataset
        if len(self.chipo.columns)>0:
            return len(self.chipo.columns)
        return -1
    
    def print_columns(self) -> None:
        # TODO Print the name of all the columns.
        print(list(self.chipo.columns))
        pass
    def most_ordered_item(self)->None:
        item_quants = self.chipo.groupby(['item_name']).agg({'quantity':'sum','order_id':'sum'})
        first_coloum = item_quants.sort_values('quantity',ascending = False)[:1]
        quantity= first_coloum.iloc[0]['quantity']
        order_id = first_coloum.iloc[0]['order_id']
        item_name = self.chipo['item_name'].mode()[0]
        return item_name, order_id,quantity  
    
    def total_item_orders(self) -> int:
       # TODO How many items were orderd in total?
       
        if self.chipo.quantity.sum() > 0:
            return self.chipo.quantity.sum()
        return -1             
    
    def total_sales(self) -> int:
        self.chipo['item_price']= self.chipo.item_price.apply(lambda x:x[1:]).astype(float)
        y=((self.chipo.item_price)*(self.chipo.quantity)).sum()
        if y>0:
            return y
        return 0.0
    
    def num_orders(self) -> int:
        # TODO
        # How many orders were made in the dataset?
        x = self.chipo["order_id"]
        y =x.max()
        if y >0:
            return y
        return -1
    
    def average_sales_amount_per_order(self) -> float:
        
        #x = total_sales(self)
        #y= num_orders(self)
        h = (((self.chipo.item_price)*(self.chipo.quantity)).sum())/(self.chipo["order_id"].max())
        if(h>0):
            return round(h,2)
        
        return 0.0
    
    def num_different_items_sold(self) -> int:
        # TODO
        # How many different items are sold?
        items = self.chipo.item_name.nunique()
        if(items >0):
            return items
        return -1
    
    def plot_histogram_top_x_popular_items(self, x:int) -> None:
        from collections import Counter
        letter_counter = Counter(self.chipo.item_name)
        # TODO
        # 1. convert the dictionary to a DataFrame
        # 2. sort the values from the top to the least value and slice the first 5 items
        # 3. create a 'bar' plot from the DataFrame
        # 4. set the title and labels:
        #     x: Items
        #     y: Number of Orders
        #     title: Most popular items
        # 5. show the plot. Hint: plt.show(block=True).
        p =pd.DataFrame.from_dict(letter_counter,orient='index',columns=['count']).reset_index()
        outputdata = p.sort_values(by='count',ascending=False).head(5)
        answer = sns.barplot(x=outputdata['index'],y=p['count'])
        answer.set(xlabel='Items', ylabel='Number of Orders',title='Most popular items')
        plt.show()
        pass
    def scatter_plot_num_items_per_order_price(self) -> None:
        variable  = lambda x: float(x[1:].strip())
        #self.chipo.item_price = self.chipo.item_price.apply(variable)
        items = self.chipo.groupby(["order_id"],as_index= False).agg({"item_price":"sum","quantity":"sum"})
        sns.scatterplot(x=items['item_price'],y=items['quantity'],s=50,color='blue')
        plt.xlabel("order price")
        plt.ylabel("Num Items")
        plt.title("Number of items per order price")
        pass
   
        

def test() -> None:
    solution = Solution()
    solution.top_x(10)
    count = solution.count()
    print(count)
    assert count == 4622
    solution.info()
    count = solution.num_column()
    assert count == 5
    item_name, order_id, quantity = solution.most_ordered_item()
    assert item_name == 'Chicken Bowl'
    assert order_id == 713926
    assert quantity == 761
    total = solution.total_item_orders()
    assert total == 4972
    assert 39237.02 == solution.total_sales()
    assert 1834 == solution.num_orders()
    assert 21.39 == solution.average_sales_amount_per_order()
    assert 50 == solution.num_different_items_sold()
    solution.plot_histogram_top_x_popular_items(5)
    solution.scatter_plot_num_items_per_order_price()

    
if __name__ == "__main__":
    # execute only if run as a script
    test()
