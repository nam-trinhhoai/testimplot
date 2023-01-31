#ifndef WellHead_H
#define WellHead_H

#include <QObject>
#include <QVector2D>

#include "idata.h"
#include "ifilebaseddata.h"

class WellBore;
class WellHeadGraphicRepFactory;

class WellHead: public IData, public IFileBasedData {
Q_OBJECT
public:
	WellHead(WorkingSetManager * workingSet,const QString &name, double X, double Y, double Z, const QString& idPath,QString date, QObject *parent =
			0);
	virtual ~WellHead();

	//IData
	virtual IGraphicRepFactory *graphicRepFactory();
	QUuid dataID() const override;
	QString name() const override{return m_name;}

	void addWellBore(WellBore *wellBore);
	void removeWellBore(WellBore *wellBore);
	QList<WellBore*> wellBores();

	double x() const;
	double y() const;
	double z() const;

	QString getDate()
	{
		return m_date;
	}

//	double displayDistance() const;
//	void setDisplayDistance(double);

	// return nullptr if could not read file
	static WellHead* getWellHeadFromDescFile(QString descFile, WorkingSetManager * workingSet, QObject *parent=0);

	QString getDirName();

signals:
	void wellBoreAdded(WellBore *wellBore);
	void wellBoreRemoved(WellBore *wellBore);
//	void displayDistanceChanged(double);
private:
	double m_x, m_y, m_z; // map coordinates

	QString m_name;
	QUuid m_uuid;
	QString m_descFile;

	QString m_date;

	QList<WellBore*> m_wellBores;
	WellHeadGraphicRepFactory * m_repFactory;
//	double m_displayDistance = 100;
};

#endif /* QTCUDAIMAGEVIEWER_QGLTEXTURE_H_ */
