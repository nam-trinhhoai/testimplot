#ifndef FOLDERDATA_H_
#define FOLDERDATA_H_

#include <QObject>
#include <QList>
#include "idata.h"

class FolderDataGraphicRepFactory;

class FolderData : public IData {
	Q_OBJECT
public:
	FolderData(WorkingSetManager * workingSet,const QString &name, QObject* parent=0);
	~FolderData();

	//IData
	virtual IGraphicRepFactory *graphicRepFactory();
	QUuid dataID() const override;
	QString name() const override{return m_name;}

	bool isDataContains(IData *data);
	void addData(IData *data);
	void removeData(IData *data);
	void deleteData(IData *data);
	QList<IData*> data();

signals:
	void dataAdded(IData *data);
	void dataRemoved(IData *data);

private:
	QString m_name;
	QUuid m_uuid;
	QList<IData*> m_content;

	FolderDataGraphicRepFactory * m_repFactory;
};

Q_DECLARE_METATYPE(FolderData*)

#endif
