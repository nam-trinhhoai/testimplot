#ifndef DatasetBlocCache_H
#define DatasetBlocCache_H

#include <QtCore/qhash.h>
#include "datasethashkey.h"
class DatasetBlocTile;

class DatasetBlocCache
{
private:
    struct Node {
	   inline Node() : keyPtr(nullptr) {

	   }

	   inline Node(DatasetBlocTile *data, size_t cost)
		   : keyPtr(nullptr), t(data), c(cost), p(nullptr), n(nullptr) {

	   }
	   const DatasetHashKey *keyPtr;
	   DatasetBlocTile *t;
	   size_t c;
	   Node *p,*n;
   };

public:
    static DatasetBlocCache* getInstance();
	~DatasetBlocCache();

	void dump();
	void releaseDatasetTiles(const std::string & path);

    void setMaxCost(size_t m);

    size_t maxCost() const { return mx; }
    size_t totalCost() const { return total; }

    int count() const { return hash.size(); }
    bool isEmpty() const { return hash.isEmpty(); }

    QList<DatasetHashKey> keys() const { return hash.keys(); }

    void clear();

    bool insert(const DatasetHashKey &key, DatasetBlocTile *object);
    DatasetBlocTile *object(const DatasetHashKey &key) const;

    bool contains(const DatasetHashKey &key) const { return hash.contains(key); }

    DatasetBlocTile *operator[](const DatasetHashKey &key) const;
    bool remove(const DatasetHashKey &key);

    DatasetBlocTile *take(const DatasetHashKey &key);
private:
    explicit DatasetBlocCache(size_t maxCost = 10e9) noexcept;

    void trim(size_t m);
    DatasetBlocTile * relink(const DatasetHashKey &key);
    void unlink(Node &n);
private:
      static DatasetBlocCache* instance;

       //Linked list management (use to update the current cache size)
       Node *f, *l;
       QHash<DatasetHashKey, Node> hash;
       size_t mx, total;
};


#endif
