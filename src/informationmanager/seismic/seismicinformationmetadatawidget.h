#ifndef SRC_INFORMATIONMANAGER_SEISMIC_SEISMICINFORMATIONMETADATAWIDGET
#define SRC_INFORMATIONMANAGER_SEISMIC_SEISMICINFORMATIONMETADATAWIDGET

#include <QWidget>

class IInformation;

class QPushButton;

class SeismicInformationMetadataWidget : public QWidget {
	Q_OBJECT
public:
	SeismicInformationMetadataWidget(IInformation* information, QWidget* parent=0);
	virtual ~SeismicInformationMetadataWidget();

public slots:
	void openDir();

private:
	IInformation* m_information;

	QPushButton* m_openButton;
	bool m_hasFolder;
	QString m_openDir;
};

#endif // SRC_INFORMATIONMANAGER_NURBS_NURBINFORMATIONMETADATAWIDGET
