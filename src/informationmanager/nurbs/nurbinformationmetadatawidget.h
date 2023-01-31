#ifndef SRC_INFORMATIONMANAGER_NURBS_NURBINFORMATIONMETADATAWIDGET
#define SRC_INFORMATIONMANAGER_NURBS_NURBINFORMATIONMETADATAWIDGET

#include <QWidget>

class IInformation;

class QPushButton;

class NurbInformationMetadataWidget : public QWidget {
	Q_OBJECT
public:
	NurbInformationMetadataWidget(IInformation* information, QWidget* parent=0);
	virtual ~NurbInformationMetadataWidget();

public slots:
	void openDir();

private:
	IInformation* m_information;

	QPushButton* m_openButton;
	bool m_hasFolder;
	QString m_openDir;
};

#endif // SRC_INFORMATIONMANAGER_NURBS_NURBINFORMATIONMETADATAWIDGET
