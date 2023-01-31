
#ifndef __ISOHORIZONATTRIBUTINFO__
#define __ISOHORIZONATTRIBUTINFO__

#include <QWidget>
#include <QString>


class IsoHorizonAttributInfo : public QWidget{
	Q_OBJECT
public:
	IsoHorizonAttributInfo(QString path, QString name, QString attributDirPath, QWidget* parent = 0);
	virtual ~IsoHorizonAttributInfo();

private:
	QString m_name = "";
	QString m_path = "";
	QString m_attributType = "";
	QString m_fullPath00000 = "";
	QString m_attributDirPath = "";
	void deleteAttribut();



private slots:
		void trt_delete();
};


#endif
