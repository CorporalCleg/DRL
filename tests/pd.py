import pandas as pd


# def length_stats(phrase):
#     phrase = phrase.lower()
#     dictionary = set(phrase)
#     print(dictionary)
#     for letter in dictionary:
#         if letter.isalpha() or letter == ' ':
#             pass
#         else:
#             phrase = phrase.replace(letter, '')
            
#     phrase = phrase.split()
#     table = pd.Series(data=phrase)
#     table = sorted(table.unique())
#     table = pd.Series(data=[len(word) for word in table], index=table)
#     return table

# print(length_stats('Мама мыла раму'))

# def cheque(product_list, **kwargs):

#     product_list = product_list[kwargs.keys()]
#     numbers = pd.Series(kwargs)
#     df = pd.DataFrame({'product': product_list.index, 'price': product_list.values})
#     df['number'] = df['product'].map(numbers)
#     df = df.sort_values(by=['product']).reset_index(drop=True)
#     df['cost'] = df['number'] * df['price']

#     return df

# products = ['bread', 'milk', 'soda', 'cream']
# prices = [37, 58, 99, 72]
# price_list = pd.Series(prices, products)
# result = cheque(price_list, soda=3, milk=2, cream=1)
# print(result)
# def get_long(data, min_length=5):
#     return data[data > min_length - 1]

# data = pd.Series([3, 5, 6, 6], ['мир', 'питон', 'привет', 'яндекс'])
# filtered = get_long(data)
# print(data)
# print(filtered)

# def best(journal):
#     filtered = journal[journal['maths'] > 3][journal['physics'] > 3][journal['computer science'] > 3]
#     return filtered

def need_to_work_better(journal):
    filtered = journal[(journal['maths'] == 2) | (journal['physics'] == 2) | (journal['computer science'] == 2)]
    return filtered
columns = ['name', 'maths', 'physics', 'computer science']
data = {
    'name': ['Иванов', 'Петров', 'Сидоров', 'Васечкин', 'Николаев'],
    'maths': [5, 4, 5, 2, 4],
    'physics': [4, 4, 4, 5, 5],
    'computer science': [5, 2, 5, 4, 3]
}
journal = pd.DataFrame(data, columns=columns)
filtered = need_to_work_better(journal)
print(journal)
print(filtered)


