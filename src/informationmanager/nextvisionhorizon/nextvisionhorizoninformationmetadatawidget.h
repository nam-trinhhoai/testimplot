#ifndef SRC_INFORMATIONMANAGER_NEXTVISIONHORIZON_NEXTVISIONHORIZONINFORMATIONMETADATAWIDGET
#define SRC_INFORMATIONMANAGER_NEXTVISIONHORIZON_NEXTVISIONHORIZONINFORMATIONMETADATAWIDGET

#include <QWidget>

class IInformation;

class QPushButton;

class NextvisionHorizonInformationMetadataWidget : public QWidget {
	Q_OBJECT
public:
	NextvisionHorizonInformationMetadataWidget(IInformation* information, QWidget* parent=0);
	virtual ~NextvisionHorizonInformationMetadataWidget();

public slots:
	void openDir();

private:
	IInformation* m_information;

	QPushButton* m_openButton;
	bool m_hasFolder;
	QString m_openDir;
};

#endif // SRC_INFORMATIONMANAGER_NURBS_NURBINFORMATIONMETADATAWIDGET
