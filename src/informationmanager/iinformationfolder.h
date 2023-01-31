#ifndef SRC_INFORMATIONMANAGER_IINFORMATIONFOLDER
#define SRC_INFORMATIONMANAGER_IINFORMATIONFOLDER

class IInformationFolder {
public:
	IInformationFolder() {}
	virtual ~IInformationFolder() {};

	virtual QString folder() const = 0;
	virtual QString mainPath() const = 0;
};

#endif
