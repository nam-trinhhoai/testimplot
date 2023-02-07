#include <QApplication>
#include <QObject>
#include <QFile>
#include <QSurfaceFormat>
#include <QTextStream>
#include <iostream>
#include <boost/filesystem/path.hpp>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "colortableregistry.h"

#include <QQmlEngine>
#include "volumeboundingmesh.h"
#include "surfacemesh.h"
#include "lookuptable.h"
#include "cudargbimage.h"
#include "cudaimagepaletteholder.h"
#include "NextMainWindow.h"
#include "cameracontroller.h"

#include "basemapwindow.h"
#include "slicerwindow.h"
#include "gdal.h"
#include "mtlengthunit.h"
#include "qmlenumwrappers.h"
#include "icomputationoperator.h"

#include "cuda_common_helpers.h"
#include "qt3dressource.h"
#include "globalconfig.h"
#include <QQuickStyle>
#include <QSplashScreen>
#include <QPixmap>
#include <QTimer>
//#include <QGLFormat>
#include <QCommandLineParser>
#include <QCommandLineOption>
#include <QFileInfo>
#include <QFontDatabase>
#include <QDebug>

namespace fs = boost::filesystem;
//
//void saveRGBDemo()
//{
//	int width=211;
//	int height=321;
//	int numBands=3;
//	GDALDataType type=GDT_Byte;
////	for(int i=0;i<GDALGetDriverCount();i++)
////	{
////		std::cout<<std::string(GDALGetDriverShortName(GDALGetDriver(i)))<<std::endl;
////	}
//	GDALDataset* poDataset = (GDALDataset*)GDALCreate( GDALGetDriverByName("GTiff"), "/home/a/demo.tif", width, height, 3, type, NULL );
//	int offset = GDALGetDataTypeSizeBytes(type);
//
//	unsigned char * data=new unsigned char[width*height*3];
//	for(unsigned int j=0;j<height;j++)
//	{
//		for(unsigned int i=0;i<width;i++)
//		{
//			if(i<100 && j<100)
//			{
//				data[3*j*width+3*i]=255;
//				data[3*j*width+3*i+1]=255;
//				data[3*j*width+3*i+2]=0;
//			}else if(j<100)
//			{
//				data[3*j*width+3*i]=255;
//				data[3*j*width+3*i+1]=0;
//				data[3*j*width+3*i+2]=0;
//			}else
//			{
//
//				data[3*j*width+3*i]=0;
//				data[3*j*width+3*i+1]=0;
//				data[3*j*width+3*i+2]=255;
//			}
//		}
//	}
//
//	poDataset->RasterIO(GF_Write, 0, 0, width, height,
//					(void*) data, width, height,
//					type, numBands, nullptr,
//					numBands * offset, numBands * offset * width, offset);
//
//	GDALClose(poDataset);
//}

class ApplicationWithExceptionCatchedInNotify : public QApplication
{
public:
    ApplicationWithExceptionCatchedInNotify(int &argc, char *argv[]) :
        QApplication(argc,argv)
    {} 
    

    bool notify(QObject* receiver, QEvent *e) override
    {
        try {
            return QApplication::notify(receiver, e);
        }
        catch(std::runtime_error e)
        {
            qDebug() << "std::runtime_error: " ;
            qDebug() << e.what();
        }
        catch(std::exception e)
        {
            qDebug() << "std::exception: ";
            qDebug() << e.what();
        }
        catch(...)
        {
            qDebug() << "exception thread : ";
        }
        return false;
    }
};

void sleep(int ms) {
	struct timespec ts = { ms / 1000, (ms % 1000) * 1000 * 1000 };
	nanosleep(&ts, NULL);
}

void updateDirProjects(QString dirProjectsFile) {
	std::vector<std::pair<QString, QString>> dirProjects;
	if (dirProjectsFile.isNull() || dirProjectsFile.isEmpty()) {
		qDebug() << "Invalid dirProjectsFile variable. Variable is empty";
		return;
	}
	QFileInfo info(dirProjectsFile);
	if (!info.exists()) {
		qDebug() << "Invalid dirProjectsFile variable. File "<< dirProjectsFile <<" does not exist.";
		return;
	} else if(!info.isReadable()) {
		qDebug() << "Invalid dirProjectsFile variable. File "<< dirProjectsFile <<" is not readable.";
		return;
	}

	QFile file(dirProjectsFile);
	if (file.open(QIODevice::ReadOnly)) {
		QTextStream stream(&file);
		QString line;
		QStringList values;
		bool test = true;

		while (stream.readLineInto(&line)) {
			values = line.split("\n")[0].split(",");
			test = test && values.count()==2;
			if (values.count()==2) {
				std::pair<QString, QString> pair(values[0], values[1] + "/");
				dirProjects.push_back(pair);
			} else if (values.count()!=0) {
				qDebug() << "Could not parse line" << line;
			}
		}


		if (!test && dirProjects.size()==0) {
			qDebug() << "Invalid dirProjectsFile variable. File "<< dirProjectsFile <<" could not be parse correctly (expected format : name,path).";
			dirProjects.clear();
			return;
		}

	} else {
		qDebug() << "Invalid dirProjectsFile variable. File "<< dirProjectsFile <<" could not be opened for reading.";
		return;
	}
	GlobalConfig& config = GlobalConfig::getConfig();
	config.setDirProjects(dirProjects);
}

void updateSessionPrefix(QString sessionPrefix) {
	GlobalConfig& config = GlobalConfig::getConfig();
	config.setSessionPathPrefix(sessionPrefix);
}

void updateSessionPath(QString sessionPath) {
	GlobalConfig& config = GlobalConfig::getConfig();
	config.setSessionPath(sessionPath);
}

void updateDatabasePath(QString databasePath) {
	GlobalConfig& config = GlobalConfig::getConfig();
	config.setDatabasePath(databasePath);
}

void updateKnownHostsFiles(const QStringList& knownHostsFiles) {
	GlobalConfig& config = GlobalConfig::getConfig();
	config.setCustomKnownHostsFiles(knownHostsFiles);
}

void updateGeneticPlotPath(const QString& geneticPlotPath) {
	GlobalConfig& config = GlobalConfig::getConfig();
	config.setGeneticPlotDirPath(geneticPlotPath);
}

void updateGeneticShiftPath(const QString& geneticShiftPath) {
	GlobalConfig& config = GlobalConfig::getConfig();
	config.setGeneticShiftDirPath(geneticShiftPath);
}

void updateTempDirPath(const QString& tempDirPath) {
	GlobalConfig& config = GlobalConfig::getConfig();
	config.setTempDirPath(tempDirPath);
}

void updateFileExplorerProgram(const QString& fileExplorerProgram) {
	GlobalConfig& config = GlobalConfig::getConfig();
	config.setFileExplorerProgram(fileExplorerProgram);
}

void initFonts() {
	int bi = QFontDatabase::addApplicationFont(":/fonts/Roboto/Roboto-BlackItalic.ttf");
	int b = QFontDatabase::addApplicationFont(":/fonts/Roboto/Roboto-Black.ttf");
	int bdi = QFontDatabase::addApplicationFont(":/fonts/Roboto/Roboto-BoldItalic.ttf");
	int i = QFontDatabase::addApplicationFont(":/fonts/Roboto/Roboto-Italic.ttf");
	int bd = QFontDatabase::addApplicationFont(":/fonts/Roboto/Roboto-Bold.ttf");
	int li = QFontDatabase::addApplicationFont(":/fonts/Roboto/Roboto-LightItalic.ttf");
	int l = QFontDatabase::addApplicationFont(":/fonts/Roboto/Roboto-Light.ttf");
	int m = QFontDatabase::addApplicationFont(":/fonts/Roboto/Roboto-Medium.ttf");
	int mi = QFontDatabase::addApplicationFont(":/fonts/Roboto/Roboto-MediumItalic.ttf");
	int r = QFontDatabase::addApplicationFont(":/fonts/Roboto/Roboto-Regular.ttf");
	int ti = QFontDatabase::addApplicationFont(":/fonts/Roboto/Roboto-ThinItalic.ttf");
	int t = QFontDatabase::addApplicationFont(":/fonts/Roboto/Roboto-Thin.ttf");

}

int main(int argc, char *argv[]) {
int retVal=0;
	try {
		GDALAllRegister();


		qputenv("QT3D_RENDERER", "opengl");

	   QSurfaceFormat format = QSurfaceFormat::defaultFormat();
	   qDebug()<<" OPENGL  :"<<format.majorVersion()<<"."<<format.minorVersion();

	   // format.setSurfaceType(QSurface::OpenGLSurface);
/*		format.setMajorVersion(3);//4
		format.setMinorVersion(0);//3
		format.setDepthBufferSize(24);
		format.setSamples(0);       //Was 4, set to 0 until we figure out how to turn of antialiasing in qml
		format.setStencilBufferSize(8);
		format.setProfile(QSurfaceFormat::CoreProfile);

	//	QGLFormat fmt;
	//	fmt.setVersion(4,3);
	//	QGLFormat::setDefaultFormat(fmt);

		  qDebug()<<" OPENGL  :"<<format.majorVersion()<<"."<<format.minorVersion();
		 QSurfaceFormat::setDefaultFormat(format);*/
		QApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
		QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
		QApplication::setWheelScrollLines(1);

		ApplicationWithExceptionCatchedInNotify app(argc, argv);

		app.setWindowIcon(QIcon(":/slicer/icons/logoIcon.png"));

		// splash screen
//		QPixmap pixmap(":/slicer/icons/logo.png");
//		QSplashScreen splash(pixmap);
//		splash.showMessage("Loading....");
//		splash.show();
		app.processEvents();

		fs::path full_path(
				QGuiApplication::applicationFilePath().toStdString());

		std::string appDirectory = full_path.parent_path().string();

		std::string palettePath = appDirectory + "/palettes";

		QFile f(":/qdarkstyle/style.qss");
		if (!f.exists()) {
			printf("Unable to set stylesheet, file not found\n");
		} else {
			f.open(QFile::ReadOnly | QFile::Text);
			QTextStream ts(&f);
			app.setStyleSheet(ts.readAll());
		}

		ColorTableRegistry::PALETTE_REGISTRY().build(palettePath);

		qmlRegisterType<Qt3DRessource>("RGTSeismicSlicer", 1, 0, "Qt3DRessource");

        // new camera controller
        qmlRegisterType<CameraController>("RGTSeismicSlicer", 1, 0, "CameraController");

		// units
		qmlRegisterType<MtLengthUnitWrapperQML>("RGTSeismicSlicer", 1, 0, "MtLengthUnitWrapperQML");//, "Cannot create object from MtLengthUnitWrapperQML class in QML");
		qmlRegisterSingletonType<MtLengthUnitWrapperQML>("RGTSeismicSlicer", 1, 0, "MtLengthUnitMETRE", MtLengthUnitWrapperQML::getMetre);
		qmlRegisterSingletonType<MtLengthUnitWrapperQML>("RGTSeismicSlicer", 1, 0, "MtLengthUnitFEET", MtLengthUnitWrapperQML::getFeet);
		qmlRegisterUncreatableMetaObject(QMLEnumWrappers::staticMetaObject, "RGTSeismicSlicer", 1, 0, "QMLEnumWrappers", "Namespace wrapper for accessing enums in QML");

		// https://stackoverflow.com/questions/60915460/call-c-function-from-qml-js-with-c-object-as-argument
		QMetaType::registerConverter<QObject*,MtLengthUnitWrapperQML>( [] (QObject* qObjPtr) {
			MtLengthUnitWrapperQML* dataPtr = qobject_cast<MtLengthUnitWrapperQML*>( qObjPtr );
			return (dataPtr == nullptr) ? MtLengthUnitWrapperQML() : MtLengthUnitWrapperQML( *dataPtr ) ;
		});

		qRegisterMetaType<IComputationOperator*>();

		cudaDeviceReset();
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);

		std::cout << "Max Texture 2D Size:" << prop.maxTexture2D[0] << "\t"
				<< prop.maxTexture2D[1] << std::endl;
		std::cout << "Max Texture 3D Size:" << prop.maxTexture3D[0] << "\t"
				<< prop.maxTexture3D[1] << "\t" << prop.maxTexture3D[2]
				<< std::endl;

		std::cout << "Max Thread per block:" << prop.maxThreadsPerBlock
				<< std::endl;

		std::cout << "Max sizes per block:" << prop.maxThreadsDim[0] << "\t"
				<< prop.maxThreadsDim[1] << "\t" << prop.maxThreadsDim[2]
				<< std::endl;
		std::cout << "Max sizes per grid:" << prop.maxGridSize[0] << "\t"
				<< prop.maxGridSize[1] << "\t" << prop.maxGridSize[2]
				<< std::endl;

		std::cout << "Managed memory:" << prop.managedMemory << std::endl;
		std::cout << "Global memory:" << prop.totalGlobalMem << std::endl;
//
//		SlicerWindow mainWindow;
//
//		if (argc > 2) {
//			std::cout << "Loading seismic:" << argv[1] << std::endl;
//			std::cout << "Loading rgt:" << " " << argv[2] << std::endl;
//			bool b = false;
//			if (b > 3)
//				b = (std::string(argv[3]) == "1");
//
//			mainWindow.open(argv[1], argv[2], b);
//		} else if (argc > 1) {
//			mainWindow.openSismageProject(argv[1]);
//		}
//	else
//		mainWindow.open("/home/a/seismic3d.xt","/home/a/rgt.xt",true);
		initFonts();

	    QCoreApplication::setOrganizationName("TOTAL");
	    QCoreApplication::setApplicationName("NextVision");

	    QCommandLineParser parser;
	    QCommandLineOption dirProjectsOption({"f" , "dir-projects-file"},
	    						QCoreApplication::translate("main", "custom file for dir projects initialization"), "file" );
	    parser.addOption(dirProjectsOption);
	    QCommandLineOption sessionPathPrefixOption({"p" , "session-path-prefix"},
	    						QCoreApplication::translate("main", "custom directory to search sessions directories from"), "directory" );
	    parser.addOption(sessionPathPrefixOption);
	    QCommandLineOption sessionPathOption({"s" , "session-path"},
	    						QCoreApplication::translate("main", "custom directory to load sessions from. session-path-prefix will be inefficient is not empty. Default empty string"), "directory" );
	    parser.addOption(sessionPathOption);
	    QCommandLineOption databasePathOption({"d" , "database-path"},
	    						QCoreApplication::translate("main", "custom directory to load databases cache from"), "directory" );
	    parser.addOption(databasePathOption);
	    QCommandLineOption knownHostsPathOption(QStringList() << "known-hosts",
	    						QCoreApplication::translate("main", "custom known hosts files to search hosts from"), "knownHostsFiles" );
	    parser.addOption(knownHostsPathOption);
	    QCommandLineOption geneticPlotPathOption(QStringList() << "genetic-plot-path",
	    						QCoreApplication::translate("main", "directory used to prefix plots from genetic algorithm"), "geneticPlotPath" );
	    parser.addOption(geneticPlotPathOption);
	    QCommandLineOption geneticShiftPathOption(QStringList() << "genetic-shift-path",
	    						QCoreApplication::translate("main", "directory used to prefix shifts from genetic algorithm"), "geneticShiftPath" );
	    parser.addOption(geneticShiftPathOption);
	    QCommandLineOption tempPathOption(QStringList() << "temp-dir-path",
	    						QCoreApplication::translate("main", "directory used to stored temporary files"), "tempDirPath" );
	    parser.addOption(tempPathOption);
	    QCommandLineOption fileExplorerProgramOption(QStringList() << "file-explorer-program",
	    						QCoreApplication::translate("main", "program to explore files"), "fileExplorerProgram" );
	    parser.addOption(fileExplorerProgramOption);
	    parser.process(app);
	    if(parser.isSet(dirProjectsOption)) {
        	QString dirProjectsFile = parser.value(dirProjectsOption);
        	updateDirProjects(dirProjectsFile);
        }
	    if(parser.isSet(sessionPathPrefixOption)) {
        	QString sessionPathPrefix = parser.value(sessionPathPrefixOption);
        	updateSessionPrefix(sessionPathPrefix);
        }
	    if(parser.isSet(sessionPathOption)) {
        	QString sessionPath= parser.value(sessionPathOption);
        	updateSessionPath(sessionPath);
        }
	    if(parser.isSet(databasePathOption)) {
        	QString databasePath = parser.value(databasePathOption);
        	updateDatabasePath(databasePath);
        }
	    if(parser.isSet(knownHostsPathOption)) {
        	QStringList knownHostsFiles = parser.values(knownHostsPathOption);
        	updateKnownHostsFiles(knownHostsFiles);
        }
	    if(parser.isSet(geneticPlotPathOption)) {
        	QString geneticPlotPath = parser.value(geneticPlotPathOption);
        	updateGeneticPlotPath(geneticPlotPath);
        }
	    if(parser.isSet(geneticShiftPathOption)) {
        	QString geneticShiftPath = parser.value(geneticShiftPathOption);
        	updateGeneticShiftPath(geneticShiftPath);
        }
	    if(parser.isSet(tempPathOption)) {
        	QString tempDirPath = parser.value(tempPathOption);
        	updateTempDirPath(tempDirPath);
        }
	    if (parser.isSet(fileExplorerProgramOption)) {
	    	QString fileExplorerProgram = parser.value(fileExplorerProgramOption);
	    	updateFileExplorerProgram(fileExplorerProgram);
	    }



		NextMainWindow nextVisionMainWindow(nullptr);
        nextVisionMainWindow.show();
//		splash.finish(&nextVisionMainWindow);

		QTimer t;
		t.setInterval(300);
		t.setSingleShot(true);
		QObject::connect(&t, SIGNAL(timeout()), &nextVisionMainWindow, SLOT(show()));
//		QObject::connect(&t, SIGNAL(timeout()), &splash, SLOT(close()));
		t.start();
		retVal = app.exec();
	} catch (...) {
		std::cerr << "An exception occured " << std::endl;
	}
	return retVal;
}




