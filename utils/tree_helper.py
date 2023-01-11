from tqdm import tqdm

class TreeHelper(object):

    def __init__(self, dir_tree) -> None:
        self.dir_tree = dir_tree

    def level_tree_ids(self, level_tree):
        level_tree_ids = {}
        level_tree_item = {}
        for k_t, v_t in level_tree.items():
            level_items = {}
            level_ids = {}

            for i, item in enumerate(v_t):
                level_items[i] = item
                level_ids[item] = i
            
            
            level_tree_item[k_t] = level_items
            level_tree_ids[k_t] = level_ids
        
        return level_tree_item, level_tree_ids


    def load_tree(self):
        tree_level = {}
        level_tree = {}

        parent2child = {}

        with open(self.dir_tree, "r") as dr:
            progress_tree = tqdm(dr)
            for i, line in enumerate(progress_tree):
                line = line[:-1].lower()
                category = line.split(" > ")[-1]
                category_all = line.split(" > ")

                # if len(category_all) > 1:
                #     for i_cat, cat in enumerate(category_all):
                #         if i_cat > 0:
                #             if category_all[i_cat - 1] not in parent2child:
                #                 parent2child[category_all[i_cat - 1]] = set()
                #             else:
                #                 parent2child[category_all[i_cat - 1]].add(cat)

                for i, cat in enumerate(category_all):
                    if i > 0:
                        parent = category_all[i - 1]
                        try:
                            parent2child[parent].add(cat)
                        except:
                            parent2child[parent] = set()
                            parent2child[parent].add(cat)
                
                level = len(line.split(" > "))
                if category not in tree_level:
                    tree_level[category] = level
                    if level not in level_tree:
                        level_tree[level] = []
                    level_tree[level] += [category]

        level_tree_item, level_tree_ids = self.level_tree_ids(level_tree)

        return tree_level, level_tree, parent2child, level_tree_item, level_tree_ids