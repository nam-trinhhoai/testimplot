
#ifndef __NEXTVISIONHORIZONATTRIBUTINFO__
#define __NEXTVISIONHORIZONATTRIBUTINFO__

#include <QWidget>
#include <QString>

class WorkingSetManager;


class NextvisionHorizonAttributInfo : public QWidget{
	Q_OBJECT
public:
	NextvisionHorizonAttributInfo(QString path, QString name, WorkingSetManager *workingSetManager, QString horizonName, QString horizonPath, QWidget* parent = 0);
	virtual ~NextvisionHorizonAttributInfo();

private:
	QString m_name = "";
	QString m_path = "";
	QString m_attributType = "";
	QString m_nameWithoutExt = "";
	void deleteAttribut();
	WorkingSetManager *m_workingSetManager = nullptr;
	QString m_horizonName = "";
	QString m_horizonPath = "";



private slots:
		void trt_delete();
};


#endif
