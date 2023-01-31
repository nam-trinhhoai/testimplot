#ifndef DatasetHashKey_H
#define DatasetHashKey_H

#include <string>
#include <QHashFunctions>
class  DatasetHashKey {
public:
	DatasetHashKey(const std::string &path, int d0, int d1);
	DatasetHashKey(const DatasetHashKey &tc);
	~DatasetHashKey(){}

	void dump()const;

	DatasetHashKey & operator= ( const DatasetHashKey & val );
	bool operator==(const DatasetHashKey &lhs) const;

	std::string hashKey() const;

	std::string path() const{return m_path;}

private:
	std::string m_path;
	int m_d0;
	int m_d1;
};

//Needed for our hash maps!
inline uint qHash(const DatasetHashKey& k) {
    return qHash(QString(k.hashKey().c_str()));
}


#endif
