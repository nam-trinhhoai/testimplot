
#ifndef __FIXEDLAYERIMPLFREEHORIZONFROMDATASETANDCUBE__
#define __FIXEDLAYERIMPLFREEHORIZONFROMDATASETANDCUBE__


#include <QString>


#include <fixedlayersfromdatasetandcube.h>

class FixedLayerImplFreeHorizonFromDatasetAndCube : public FixedLayersFromDatasetAndCube//, public IData, public StackableData, public IsoChronProvider
{
public:
	FixedLayerImplFreeHorizonFromDatasetAndCube(QString dirPath, QString dirName, QString seismicName, WorkingSetManager *workingSet,
						const FixedLayersFromDatasetAndCube::Grid3DParameter &params, QObject *parent = 0);

	virtual ~FixedLayerImplFreeHorizonFromDatasetAndCube();
	void getImageForIndex(long newIndex,
			CUDAImagePaletteHolder* attrCudaBuffer, CUDAImagePaletteHolder* isoCudaBuffer);
	// use rgb1
	bool getImageForIndex(long newIndex,
			QByteArray& attrBuffer, QByteArray& isoBuffer);

	void setCurrentImageIndexInternal(long newIndex);
	void setCurrentImageIndex(long newIndex);
	QString getIsoFileFromIndex(int index);
	QColor getHorizonColor();
	QString dirPath() { return m_dirPath; }
	void setHorizonColor(QColor col);
	QString getObjFile(int index) const	override { return ""; }
	bool enableSlicePropertyPanel() override { return false; }

	bool enableScaleSlider() override;
	int getNbreGccScales() override;
	QString propertyPanelType() override { return "freehorizon"; }
	void setGccIndex(int value) override;


private:
	void loadIsoAndAttribute(QString attribute);
	QString getAttributFileFromIndex(int index);
	QString readAttributData(short *data, long size);


	QString m_dirPath = "";
	QString m_dirName = "";
	QString m_seismicName = "";
	QString m_attributName;
	QString m_isoName = "";
	int m_gccScale = 0;
	int m_gccNbreScales = -1;
	QString m_attributDataSetPath = "";
};

#endif
