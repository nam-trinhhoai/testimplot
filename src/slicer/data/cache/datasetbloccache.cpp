#include "datasetbloccache.h"
#include "datasetbloctile.h"

#include <iostream>
DatasetBlocCache* DatasetBlocCache::instance = nullptr;


DatasetBlocCache* DatasetBlocCache::getInstance()
{
    if (instance == nullptr)
    {
        instance = new DatasetBlocCache();
    }
    return instance;
}

DatasetBlocCache::DatasetBlocCache(size_t amaxCost) noexcept :
		f(nullptr), l(nullptr), mx(amaxCost), total(0) {
}

DatasetBlocCache::~DatasetBlocCache() {
	clear();
}

void DatasetBlocCache::clear() {
	while (f) {
		delete f->t;
		f = f->n;
	}
	hash.clear();
	l = nullptr;
	total = 0;
}

void DatasetBlocCache::releaseDatasetTiles(const std::string & path)
{
	QHash<DatasetHashKey, Node>::iterator i = hash.begin();
	while(i!=hash.end())
	{
		if(i.key().path()==path)
		{
			unlink(*i);
			 i = hash.begin();
			 continue;
		}
		i++;
	}
}


void DatasetBlocCache::setMaxCost(size_t m) {
	mx = m;
	trim(mx);
}

DatasetBlocTile* DatasetBlocCache::object(const DatasetHashKey &key) const {
	return const_cast<DatasetBlocCache*>(this)->relink(key);
}

DatasetBlocTile * DatasetBlocCache::operator[](const DatasetHashKey &key) const {
	return object(key);
}

bool DatasetBlocCache::remove(const DatasetHashKey &key) {
	QHash<DatasetHashKey, Node>::iterator i = hash.find(key);
	if (QHash<DatasetHashKey, Node>::const_iterator(i) == hash.constEnd()) {
		return false;
	} else {
		unlink(*i);
		return true;
	}
}

DatasetBlocTile *DatasetBlocCache::take(const DatasetHashKey &key)
{
	QHash<DatasetHashKey, Node>::iterator i = hash.find(key);
	if (i == hash.end())
	return nullptr;

	Node &n = *i;
	DatasetBlocTile *t = n.t;
	n.t = nullptr;
	unlink(n);
	return t;
}

bool DatasetBlocCache::insert(const DatasetHashKey &akey, DatasetBlocTile *aobject) {
	size_t acost=aobject->memoryCost();
	remove(akey);
	if (total + acost > mx) {
		delete aobject;
		return false;
	}
//	std::cout<<"New tile injected:"<<std::endl;
//	akey.dump();
	trim(mx - acost);

	Node sn(aobject, acost);
	QHash<DatasetHashKey, Node>::iterator i = hash.insert(akey, sn);
	total += acost;
	Node *n = &i.value();
	n->keyPtr = &i.key();
	if (f)
		f->p = n;
	n->n = f;
	f = n;
	if (!l)
		l = f;
	return true;
}

void DatasetBlocCache::trim(size_t m) {
	Node *n = l;
	while (n && total > m) {
		Node *u = n;
		n = n->p;
		unlink(*u);
	}
}

void DatasetBlocCache::unlink(Node &n) {
	if (n.p)
		n.p->n = n.n;
	if (n.n)
		n.n->p = n.p;

	if (l == &n)
		l = n.p;
	if (f == &n)
		f = n.n;
	total -= n.c;

	DatasetBlocTile *obj = n.t;
	hash.remove(*n.keyPtr);
	delete obj;
}

void DatasetBlocCache::dump()
{
	QHash<DatasetHashKey, Node>::iterator i = hash.begin();
	while(i!=hash.end())
	{
		i.key().dump();
		i++;
	}

}

DatasetBlocTile * DatasetBlocCache::relink(const DatasetHashKey &key) {
	QHash<DatasetHashKey, Node>::iterator i = hash.find(key);
	if (QHash<DatasetHashKey, Node>::const_iterator(i) == hash.constEnd())
		return nullptr;

	Node &n = *i;
	if (f != &n) {
		if (n.p)
			n.p->n = n.n;
		if (n.n)
			n.n->p = n.p;
		if (l == &n)
			l = n.p;
		n.p = nullptr;
		n.n = f;
		f->p = &n;
		f = &n;
	}
	return n.t;
}

