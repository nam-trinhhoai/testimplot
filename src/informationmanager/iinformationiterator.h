#ifndef SRC_INFORMATIONMANAGER_IINFORMATIONITERATOR_H
#define SRC_INFORMATIONMANAGER_IINFORMATIONITERATOR_H

class IInformation; // class because include would create a circular include

#include <memory>

class IInformationIterator {
public:
	IInformationIterator() {};
	virtual ~IInformationIterator() {};

	virtual bool isValid() = 0;
	virtual const IInformation* cobject() const = 0;
	virtual IInformation* object() = 0;

	virtual bool hasNext() const = 0;
	virtual bool next() = 0;

	virtual std::shared_ptr<IInformationIterator> copy() const = 0;
};

#endif // SRC_INFORMATIONMANAGER_IINFORMATIONITERATOR_H
