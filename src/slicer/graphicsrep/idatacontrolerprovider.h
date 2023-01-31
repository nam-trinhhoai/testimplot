#ifndef IDataControlerProvider_H
#define IDataControlerProvider_H

class RandomView3D;
class DataControler;

class IDataControlerProvider {
public:
	virtual ~IDataControlerProvider(){}
	virtual void setDataControler(DataControler * controler)=0;
	virtual DataControler* dataControler() const=0;
};

#endif
