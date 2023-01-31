#include "generalizationsectionwidget.h"
#include "functionselector.h"
#include "resamplespline.h"
#include "iterationlistwidgetitem.h"
#include "colortable/colortableregistry.h"
#include "CubeIO.h"

#include <kddockwidgets/DockWidget.h>

#include <QComboBox>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QSpinBox>
#include <QTreeWidget>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QPushButton>
#include <QFileDialog>
#include <QDoubleValidator>
#include <QDebug>
#include <QListWidget>
#include <QMessageBox>
#include <QInputDialog>
#include <QMenu>
#include <QAction>

#include <limits>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <memory>

/*#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

extern "C" {
#define Linux 1
#include "image.h"
#include "comOCR.h"
}


extern "C" {
    int c_wrcmim_OCR(struct image* im,struct com_OCR* comOCR);
    int c_ircmim_OCR(struct image* im,struct com_OCR* comOCR);
    struct image *image_(char *nom,char *mode,char *verif,struct nf_fmt *gfmt);
    int inr_xargc = 0;
    char** inr_xargv = NULL;
}

extern int debug_;
struct com_OCR  comOCR ;
struct com_OCR  *pcomOCR ;*/

long GeneralizationSectionWidget::windowNextId = 0;

GeneralizationSectionWidget::GeneralizationSectionWidget(QWidget *parent) :
	KDDockWidgets::MainWindow("Generalize section "+QString::number(windowNextId++), KDDockWidgets::MainWindowOption_None, parent),
    fsWatcher(this)
{
    m_uniqueName = "Generalize section "+QString::number(windowNextId-1);
    setupGui();

    connect(m_suffixLineEdit, &QLineEdit::textChanged, this, &GeneralizationSectionWidget::updateSaveSuffix);
    connect(m_yStepSpinBox, SELECT<int>::OVERLOAD_OF(&QSpinBox::valueChanged), this, &GeneralizationSectionWidget::updateYStep);

    connect(m_updateButton, &QPushButton::clicked, this, &GeneralizationSectionWidget::launchGeneralization);

    m_yStepSpinBox->setMaximum(std::numeric_limits<int>::max());
    m_yStepSpinBox->setMinimum(1);

    m_sliceSlider->setRange(0,0);
    m_sliceSlider->setValue(0);

    connect(m_orientationComboBox, &QComboBox::currentTextChanged, this, &GeneralizationSectionWidget::changeOrientation);

    connect(m_sliceSlider, &QSlider::valueChanged, m_sliceSpinBox, &QSpinBox::setValue);
    connect(m_sliceSpinBox, SELECT<int>::OVERLOAD_OF(&QSpinBox::valueChanged), m_sliceSlider, &QSlider::setValue);

    m_orientationComboBox->setCurrentIndex(0);

    m_checkpointListWidget->setSortingEnabled(true);

    connect(&fsWatcher, &QFileSystemWatcher::directoryChanged, this, &GeneralizationSectionWidget::updateCheckpointListWidget);
    connect(m_checkpointListWidget, &QListWidget::itemSelectionChanged, this, &GeneralizationSectionWidget::updateViewerListFromSelection);

    connect(m_predictSampleRateSpinBox, SELECT<double>::OVERLOAD_OF(&QDoubleSpinBox::valueChanged), this, &GeneralizationSectionWidget::setPredictSampleRateInternal);

    connect(m_sliceSlider, &QSlider::sliderReleased, this, &GeneralizationSectionWidget::launchGeneralization);

    m_randomComboBox->hide();

    connect(m_ijkWellListWidget, &QListWidget::currentTextChanged, this, [this](QString well_name) {
        if (well_name.isEmpty() || well_name.isNull()) {
            return;
        }

        Well well;
        well.name = well_name;
        QVector<QVector3D> trajectory;

        qDebug() << IJK_dir+"/WELL_TRAJECTORIES/"+well_name+".txt";
        QFile file(IJK_dir+"/WELL_TRAJECTORIES/"+well_name+".txt");
        bool test = false;
        if (file.open(QIODevice::ReadOnly)) {
            QTextStream stream(&file);
            QString line;
            QStringList values;
            test = true;

            while (test && stream.readLineInto(&line)) {
                values = line.split("\n")[0].split(" ");
                test = values.count()==3;
                if (test) {
                    QVector3D pt;
                    pt.setX(values[0].toDouble(&test));
                    if (test) {
                       pt.setY(values[1].toDouble(&test));
                       if (test) {
                           pt.setZ(values[2].toDouble(&test));
                           if (test) {
                               trajectory.append(pt);
                           }
                       }
                    }
                }
            }
        }

        this->wellGraphicManager.addWell(well, trajectory);

        /*while (index<(*wells).size() && (*wells)[index].name.compare(well_name)!=0) {
            index ++;
        }
        if (index < (*wells).size()) {
            this->wellGraphicManager.setWell((*wells)[index]);
        }*/

    });

    connect(m_jsonWellTreeWidget, &QTreeWidget::itemSelectionChanged, this, &GeneralizationSectionWidget::jsonWellSelectionChanged);

    connect(m_jsonWellTreeWidget->model(), &QAbstractItemModel::dataChanged, this, &GeneralizationSectionWidget::jsonWellModelDataChanged);

    /*QStringList horizontalHeaderList;
    horizontalHeaderList << "X" << "Y" << "Z";
    m_limitWidget->setHorizontalHeaderLabels(horizontalHeaderList);
    QStringList verticalHeaderList;
    verticalHeaderList << "Start" << "End";
    m_limitWidget->setVerticalHeaderLabels(verticalHeaderList);
    m_limitWidget->setRowCount(2);
    m_limitWidget->setColumnCount(3);

    xminLineEdit = new QLineEdit;
    QIntValidator* validator = new QIntValidator(0, std::numeric_limits<int>::max());
    xminLineEdit->setValidator(validator);
    connect(xminLineEdit, &QLineEdit::editingFinished, this, &GeneralizationWidget::updateXMin);
    m_limitWidget->setCellWidget(0, 0, xminLineEdit);

    xmaxLineEdit = new QLineEdit;
    validator = new QIntValidator(0, std::numeric_limits<int>::max());
    xmaxLineEdit->setValidator(validator);
    connect(xmaxLineEdit, &QLineEdit::editingFinished, this, &GeneralizationWidget::updateXMax);
    m_limitWidget->setCellWidget(1, 0, xmaxLineEdit);

    yminLineEdit = new QLineEdit;
    validator = new QIntValidator(0, std::numeric_limits<int>::max());
    yminLineEdit->setValidator(validator);
    connect(yminLineEdit, &QLineEdit::editingFinished, this, &GeneralizationWidget::updateYMin);
    m_limitWidget->setCellWidget(0, 1, yminLineEdit);

    ymaxLineEdit = new QLineEdit;
    validator = new QIntValidator(0, std::numeric_limits<int>::max());
    ymaxLineEdit->setValidator(validator);
    connect(ymaxLineEdit, &QLineEdit::editingFinished, this, &GeneralizationWidget::updateYMax);
    m_limitWidget->setCellWidget(1, 1, ymaxLineEdit);

    zminLineEdit = new QLineEdit;
    validator = new QIntValidator(0, std::numeric_limits<int>::max());
    zminLineEdit->setValidator(validator);
    connect(zminLineEdit, &QLineEdit::editingFinished, this, &GeneralizationWidget::updateZMin);
    m_limitWidget->setCellWidget(0, 2, zminLineEdit);

    zmaxLineEdit = new QLineEdit;
    validator = new QIntValidator(0, std::numeric_limits<int>::max());
    zmaxLineEdit->setValidator(validator);
    connect(zmaxLineEdit, &QLineEdit::editingFinished, this, &GeneralizationWidget::updateZMax);
    m_limitWidget->setCellWidget(1, 2, zmaxLineEdit);*/
}

GeneralizationSectionWidget::~GeneralizationSectionWidget()
{
    for (auto& e : logPalette) {
        e.first->deleteLater();
        delete e.second;
    }
    logPalette.clear();
}

// TODO
void GeneralizationSectionWidget::reset() {
    hasBeenReset = true;

    float imageRatio = 1;
    if (m_currentPredictSampleRateSet) {
        imageRatio = predictSampleRate / currentPredictSampleRate;
    }
    currentPredictSampleRate = predictSampleRate;
    m_currentPredictSampleRateSet = true;

    // delete grid
    for (auto e : viewers) {
        e.view->deleteLater();
        synchronizer.remove(e.view);
        wellGraphicManager.disconnectViewer(e.view);
    }
    viewers.clear();

    for (auto& e : seismicPalette) {
        e.first->deleteLater();
        delete e.second;
    }
    seismicPalette.clear();
    for (auto& e : logPalette) {
        e.first->deleteLater();
        delete e.second;
    }
    logPalette.clear();

    wellGraphicManager.setRatioBetweenWellsAndImage(trainSampleRate/currentPredictSampleRate);


    int baseH = dimx;
    int w = (m_orientationComboBox->currentText().compare("inline")==0) ? dimy : dimz;
    int h = std::floor((baseH-1)*seismicSampleRate/currentPredictSampleRate)+1;

    QPointF imageCenter;
    if (viewers.size()>0) {
	    QPoint imageCenterInViewport = viewers[0].view->viewport()->rect().center();
	    imageCenter = viewers[0].view->mapToScene(imageCenterInViewport);
    } else {
        imageCenter = QPoint(w/2, h/2);
    }

    for (int i=0; i<seismicNames.size(); i++) {
        // create viewer
        Viewer viewer;
        viewer.view = new MuratCanvas2dFullView;
        viewer.view->initScene();
        viewer.view->setWindowName(seismicNames[i].second.split("/").last().split(".")[1]);
        viewer.view->toggleCurve(false);
        viewer.file = seismicNames[i].second;
        viewer.isSeismic = true;
        viewer.iter_number = i - seismicNames.size();
        PaletteWidget* paletteWidget = new PaletteWidget();
        paletteWidget->hide();
        PaletteHolder* paletteHolder = new PaletteHolder(seismicDynamic[i].first, seismicDynamic[i].second);
        paletteWidget->setPaletteHolder(paletteHolder);
        paletteWidget->setLookupTable(ColorTableRegistry::DEFAULT());
        std::pair<PaletteWidget*, PaletteHolder*> pair(paletteWidget, paletteHolder);
        viewer.paletteWidget = paletteWidget;

        connect(viewer.view , &MuratCanvas2dFullView::paletteRequestedSignal, this, [this](MuratCanvas2dFullView* view) {
            PaletteWidget* paletteWidget = nullptr;
            for (auto& e: viewers) {
                if (paletteWidget!= e.paletteWidget && view != e.view) {
                    e.paletteWidget->hide();
                } else if (view == e.view) {
                    paletteWidget = e.paletteWidget;
                    e.paletteWidget->show();
                }
            }
        });

        connect(paletteWidget, &PaletteWidget::lookupTableChanged, this, [this, paletteWidget]() {
            this->updateViewerPalette(paletteWidget);
        });
        connect(paletteWidget, SELECT<float>::OVERLOAD_OF(&PaletteWidget::opacityChanged), this, [this, paletteWidget]() {
            this->updateViewerPalette(paletteWidget);
        });
        connect(paletteWidget, &PaletteWidget::rangeChanged, this, [this, paletteWidget]() {
            this->updateViewerPalette(paletteWidget);
        });

        seismicPalette.append(pair);

       // viewer.view->setWindowName(seismicNames[i].first);
        viewers.append(viewer);
        wellGraphicManager.connectViewer(viewer.view);

        QImage img(w, h, QImage::Format_Indexed8);

        if (viewers[i].mat) {
            delete viewers[i].mat;
        }
        viewers[i].mat = new Matrix2DLine<float>(h, w);
        murat::io::InputOutputCube<float>* cube_seismic = murat::io::openCube<float>(seismicNames[i].second.toStdString());
        std::vector<float> tmpBuf;
        tmpBuf.resize(w*cube_seismic->getDim().getI());
        if (m_orientationComboBox->currentText().compare("inline")==0) {
            cube_seismic->readSubVolume(0,0, sliderValueToVolumeIndex(), cube_seismic->getDim().getI(), cube_seismic->getDim().getJ(), 1, tmpBuf.data());
        } else if (m_orientationComboBox->currentText().compare("xline")==0) {
            cube_seismic->readSubVolume(0, sliderValueToVolumeIndex(), 0, cube_seismic->getDim().getI(), 1, cube_seismic->getDim().getK(), tmpBuf.data());
        } else {
            // because viewer.isSeismic is true
            #pragma omp parallel for
            for (int i=0; i<randomPts.count(); i++) {
                qDebug() << "Random read for file " << viewer.file << ", point n°" << i;
                QPoint& pt = randomPts[i];
                cube_seismic->readSubVolume(0, pt.y(), pt.x(), cube_seismic->getDim().getI(), 1, 1, tmpBuf.data()+h*i);
            }
        }

        if (cube_seismic->getDim().getI()==h) {
            memcpy(viewers[i].mat->data(), tmpBuf.data(), tmpBuf.size()*sizeof(float));
        } else {
            // if cube_seismic->getDim().getI() != h, it should be baseH
            for (int traceIndex=0; traceIndex<w;traceIndex++) {
                resampleSplineFloat(currentPredictSampleRate, seismicSampleRate, tmpBuf.data()+traceIndex*baseH, baseH, ((float*) viewers[i].mat->data())+traceIndex*h, h);
            }
        }

        delete cube_seismic;
        viewers[i].mat->transpose();

        /*long max = viewers[i].mat->getLine(0)[0];
        long min = viewers[i].mat->getLine(0)[0];
        for (int j=0; j<h; j++) {
            for (int k=0; k<w; k++) {
                if (min>viewers[i].mat->getLine(j)[k]) {
                    min = viewers[i].mat->getLine(j)[k];
                }
                if (max<viewers[i].mat->getLine(j)[k]) {
                    max = viewers[i].mat->getLine(j)[k];
                }
            }
        }*/
        float min = viewers[i].paletteWidget->getRange().x();
        float max = viewers[i].paletteWidget->getRange().y();

        int N = viewers[i].paletteWidget->getLookupTable().size()-1;
        if (N> 255) {
            N = 255;
        }

        for (int j=0; j<h; j++) {
            for (int k=0; k<w; k++) {
                signed short val = N * (viewers[i].mat->getLine(j)[k] - min) / (max - min);
                if (val <0) {
                    val = 0;
                } else if (val>N) {
                    val = N;
                }
                img.scanLine(j)[k] = (unsigned char) val;
            }
        }
        QVector<QRgb> colors;
        for (int c=0; c<=N; c++) {
            std::array<int, 4> array = viewers[i].paletteWidget->getLookupTable().getColors(c);
            colors.append(qRgba(array[0], array[1], array[2], array[3]));
        }
        img.setColorTable(colors);
        //viewer.view->hide();
        viewers[i].view->setImage(img, (Matrix2DInterface*) viewers[i].mat);

        if (viewer.dockWidget==nullptr) {
            viewers[i].dockWidget = new KDDockWidgets::DockWidget(m_uniqueName+"_"+seismicNames[i].second);
            viewers[i].dockWidget->setWindowTitle(seismicNames[i].second.split("/").last().split(".")[1]);
            viewers[i].dockWidget->setTitle(seismicNames[i].second.split("/").last().split(".")[1]);
            viewers[i].dockWidget->setOptions(KDDockWidgets::DockWidget::Option_NotClosable);
            viewers[i].dockWidget->setWidget(viewers[i].view);
        }
        if (i==0) {
            addDockWidget(viewers[i].dockWidget, KDDockWidgets::Location_OnBottom, m_paramsDockWidget);
        } else {
            viewers[0].dockWidget->addDockWidgetAsTab(viewers[i].dockWidget);
        }

        synchronizer.addCanvas2d(viewers[i].view);

        connect(viewer.view , &MuratCanvas2dFullView::requestMenu, this, [this](QPoint pos, MuratCanvas2dFullView* view) {
            QMenu menu;
            //add default menu

            QAction* actionPalette = menu.addAction("Adjust palette");

            QObject::connect(actionPalette, &QAction::triggered, [this, view](bool checked) {
                PaletteWidget* paletteWidget = nullptr;
                for (auto& e: this->viewers) {
                    if (paletteWidget!= e.paletteWidget && view != e.view) {
                        e.paletteWidget->hide();
                    } else if (view == e.view) {
                        paletteWidget = e.paletteWidget;
                        e.paletteWidget->show();
                    }
                }
            });

            QAction* actionWellPlot = menu.addAction("Adjust well plotting parameters");
            QObject::connect(actionWellPlot, &QAction::triggered, [this, view](bool checked) {
                this->wellGraphicManager.runGraphicsSettingsDialog(this);
            });

            //show the menu
            menu.exec(pos);
        });
    }

    synchronizer.simulate_gentle_zoom(1, imageRatio, imageCenter);

    if (viewers.size()>0) {
        QPointF scaledCenter(imageCenter.x(), imageCenter.y()/imageRatio);
        viewers[0].view->centerOn(scaledCenter);
    }
//    m_viewContainer->setUpdatesEnabled(true);
//    m_viewContainer->updateGeometry();
}

void GeneralizationSectionWidget::launchGeneralization() {
    if(stack.size()>0) {
        stack.clear();
        QMessageBox::information(this, tr("Process busy"), tr("The Process is busy, please wait"));
        return;
    }

    float imageRatio = 1;
    if (m_currentPredictSampleRateSet) {
        imageRatio = predictSampleRate / currentPredictSampleRate;
    }
    currentPredictSampleRate = predictSampleRate;
    m_currentPredictSampleRateSet = true;

    wellGraphicManager.setRatioBetweenWellsAndImage(trainSampleRate/currentPredictSampleRate);

    randomPts.clear();
    QString random_filename;
    int w;
    int baseH = dimx;
    int h = std::floor((baseH-1)*seismicSampleRate/currentPredictSampleRate)+1;

    if (m_orientationComboBox->currentText().compare("inline")==0 || m_orientationComboBox->currentText().compare("xline")==0) {
        w = (m_orientationComboBox->currentText().compare("inline")==0) ? dimy : dimz;
    } else {
        // Read random
        random_filename = IJK_dir + "/RANDOMS/" + m_randomComboBox->currentText();
        if (m_randomComboBox->currentText().isNull() || m_randomComboBox->currentText().isEmpty()) {
            return;
        }
        QFile file(random_filename);

        bool test = false;
        if (file.open(QIODevice::ReadOnly)) {
            QTextStream stream(&file);
            QString line;
            QStringList values;
            test = true;

            while (test && stream.readLineInto(&line)) {
                values = line.split("\n")[0].split(" ");
                test = values.count()==2;
                if (test) {
                    QPoint pt;
                    pt.setX(values[0].toInt(&test));
                    if (test) {
                       pt.setY(values[1].toInt(&test));
                       if (test && pt.x()>=0 && pt.x()<dimz && pt.y()>=0 && pt.y()<dimy) {
                           randomPts.append(pt);
                       }
                    }
                }
            }
        }
        if (!test) {
            QMessageBox::warning(this, "Invalid random line", "The selected random line : "+m_randomComboBox->currentText()+" is invalid");
            return;
        }
        w = randomPts.count();
    }

    QPointF imageCenter;
    if (viewers.size()>0) {
	    QPoint imageCenterInViewport = viewers[0].view->viewport()->rect().center();
	    imageCenter = viewers[0].view->mapToScene(imageCenterInViewport);
    } else {
        imageCenter = QPoint(w/2, h/2);
    }

    for (Viewer& viewer : viewers) {
        if (viewer.isSeismic) {
            QImage img(w, h, QImage::Format_Indexed8);

            if (viewer.mat) {
                delete viewer.mat;
            }
            viewer.mat = new Matrix2DLine<float>(h, w);
            murat::io::InputOutputCube<float>* cube_seismic = murat::io::openCube<float>(viewer.file.toStdString());
            std::vector<float> tmpBuf;
            tmpBuf.resize(w*cube_seismic->getDim().getI());
            if (m_orientationComboBox->currentText().compare("inline")==0) {
                cube_seismic->readSubVolume(0,0, sliderValueToVolumeIndex(), cube_seismic->getDim().getI(), cube_seismic->getDim().getJ(), 1, tmpBuf.data());
            } else if (m_orientationComboBox->currentText().compare("xline")==0) {
                cube_seismic->readSubVolume(0, sliderValueToVolumeIndex(), 0, cube_seismic->getDim().getI(), 1, cube_seismic->getDim().getK(), tmpBuf.data());
            } else {
				#pragma omp parallel for
                for (int i=0; i<randomPts.count(); i++) {
                    qDebug() << "Random read for file " << viewer.file << ", point n°" << i;
                    QPoint& pt = randomPts[i];
                    cube_seismic->readSubVolume(0, pt.y(), pt.x(), cube_seismic->getDim().getI(), 1, 1, tmpBuf.data()+h*i);
                }

                QString random_file = QFileInfo(viewer.file).fileName().split(".")[1]; // remove seismic3d. and .xt
                if (!cube_seismic->getNativeType().EqualsTo(murat::io::SampleType::INT16)) {
                    qDebug() << "Sample type expected is short. Cube sample type is not short.";
                }
                murat::io::InputOutputCube<short>* cube_seismic_random = murat::io::openOrCreateCube<short>((random_filename+"_"+random_file+".xt").toStdString(), h, w, 1, 0, cube_seismic->getNativeType(), true);
                short* buf = new short[h*w];
                float* buf_float = (float*) viewer.mat->data();
#pragma omp parallel for schedule(static)
                for (long k=0; k<h*w; k++) {
                    buf[k] = (short) (buf_float[k]);
                }
                cube_seismic_random->writeSubVolume(0, 0, 0, buf, h, w, 1);
                delete buf;
                delete cube_seismic_random;
            }

            if (cube_seismic->getDim().getI()==h) {
                memcpy(viewer.mat->data(), tmpBuf.data(), tmpBuf.size()*sizeof(float));
            } else {
                for (int traceIndex=0; traceIndex<w;traceIndex++) {
                    resampleSplineFloat(currentPredictSampleRate, seismicSampleRate, tmpBuf.data()+traceIndex*baseH, baseH, ((float*) viewer.mat->data())+traceIndex*h, h);
                }
            }

            delete cube_seismic;
            viewer.mat->transpose();

            // Get dynamic index
            int i = 0;
            while (i<seismicNames.size() && seismicNames[i].second.compare(viewer.file)!=0) {
                i++;
            }

            float min = viewer.paletteWidget->getRange().x();
            float max = viewer.paletteWidget->getRange().y();

            int N = viewer.paletteWidget->getLookupTable().size() - 1;
            if (N>255) {
                N = 255;
            }
            QVector<QRgb> colors;
            for (int c=0; c<=N; c++) {
                std::array<int, 4> array = viewer.paletteWidget->getLookupTable().getColors(c);
                colors.append(qRgba(array[0], array[1], array[2], array[3]));
            }

            img.setColorTable(colors);

            for (int j=0; j<h; j++) {
                for (int k=0; k<w; k++) {
                    signed short val = N * (viewer.mat->getLine(j)[k] - min) / (max - min);
                    if (val <0) {
                        val = 0;
                    } else if (val>N) {
                        val = N;
                    }
                    img.scanLine(j)[k] = (unsigned char) val;
                }
            }

            viewer.view->setImage(img, (Matrix2DInterface*) viewer.mat);
        } else {
            int h = std::floor((baseH-1)*seismicSampleRate/currentPredictSampleRate)+1;
            QImage img(w, h, QImage::Format_Grayscale8);
            if (viewer.mat) {
                delete viewer.mat;
            }
            viewer.mat = new Matrix2DLine<float>(w, h);
            img.fill(0);
            memset(viewer.mat->data(), 0, w*h*sizeof(float));

            viewer.view->setImage(img, (Matrix2DInterface*) viewer.mat);
            stack.push_back(viewer.file);
        }
    }
    wellGraphicManager.setSlice(sliderValueToVolumeIndex());
    wellGraphicManager.setRandomLine(randomPts);
    synchronizer.simulate_gentle_zoom(1, imageRatio, imageCenter);

    if (viewers.size()>0) {
        QPointF scaledCenter(imageCenter.x(), imageCenter.y()/imageRatio);
        viewers[0].view->centerOn(scaledCenter);
    }

    predictFromStack();

    /*if (m_ijkWellListWidget->count()==0) {
    	findIJKDirectory();
        m_ijkWellListWidget->addItem("");
        QDir dir(IJK_dir+"/WELL_TRAJECTORIES");
        if (dir.exists()) {
            QFileInfoList infoList = dir.entryInfoList(QStringList() << "*.txt", QDir::Files);
            for (QFileInfo& info : infoList) {
                trajectories.append(info.baseName());
                m_ijkWellListWidget->addItem(info.baseName());
            }
        } else {
            QMessageBox::warning(this, tr("Well trajectories missing"), tr("Please, run the IJK program."));
        }
    }*/

/*    auto t1 = std::chrono::steady_clock::now();
    if (seismicNames.size()>0 && logNames.size()>0) {
        int slice = m_sliceSlider->value();

        int ymin, ymax, zmin, zmax;
        if (m_orientationComboBox->currentText().compare("inline")==0) {
            ymin = 0;
            ymax = dimy-1;
            zmin = slice;
            zmax = slice;
        } else if (m_orientationComboBox->currentText().compare("xline")==0) {
            ymin = slice;
            ymax = slice;
            zmin = 0;
            zmax = dimz-1;
        } else {
            qDebug() << "GeneralizationWidget::launchGeneralization unexpected combobox input";
        }
        emit generalize(0, dimx-1, ymin, ymax, zmin, zmax, che);

        // fill the grid layout
        if (hasBeenReset) {


            hasBeenReset = false;
        } else {
            int h = dimx;
            int w = (m_orientationComboBox->currentText().compare("inline")==0) ? dimy : dimz;
            for (int i=0; i<seismicNames.size(); i++) {
                QImage img(w, h, QImage::Format_Grayscale8);

                viewers[i].mat->transpose();
                InputOutputCube<float>* cube_seismic = openCube<float>(seismicNames[i].second.toStdString());
                if (m_orientationComboBox->currentText().compare("inline")==0) {
                    cube_seismic->readSubVolume(0,0, m_sliceSlider->value(), cube_seismic->getDim().getI(), cube_seismic->getDim().getJ(), 1, (float*) viewers[i].mat->data());
                } else {
                    cube_seismic->readSubVolume(0, m_sliceSlider->value(), 0, cube_seismic->getDim().getI(), 1, cube_seismic->getDim().getK(), (float*) viewers[i].mat->data());
                }
                delete cube_seismic;
                viewers[i].mat->transpose();

                /*long max = viewers[i].mat->getLine(0)[0];
                long min = viewers[i].mat->getLine(0)[0];
                for (int j=0; j<h; j++) {
                    for (int k=0; k<w; k++) {
                        if (min>viewers[i].mat->getLine(j)[k]) {
                            min = viewers[i].mat->getLine(j)[k];
                        }
                        if (max<viewers[i].mat->getLine(j)[k]) {
                            max = viewers[i].mat->getLine(j)[k];
                        }
                    }
                }* /
                float min = seismicDynamic[i].first;
                float max = seismicDynamic[i].second;

                for (int j=0; j<h; j++) {
                    for (int k=0; k<w; k++) {
                        signed short val = 255 * (viewers[i].mat->getLine(j)[k] - min) / (max - min);
                        if (val <0) {
                            val = 0;
                        } else if (val>255) {
                            val = 255;
                        }
                        img.scanLine(j)[k] = (unsigned char) val;
                    }
                }
                viewers[i].view->hide();
                viewers[i].view->setImage(img, (Matrix2DInterface*) viewers[i].mat);
            }
        }
    }
    compute_t1 = std::chrono::steady_clock::now();
    initialDisplay = std::chrono::duration_cast<std::chrono::milliseconds>(compute_t1 - t1).count();
*/
}

void GeneralizationSectionWidget::updateSaveSuffix(QString s) {
    saveSuffix = s;
    emit saveSuffixChanged(saveSuffix);
}

/*void GeneralizationWidget::updateXMin() {
    bool ok;
    int i = xminLineEdit->text().toInt(&ok);
    if (!ok) {
        i = 0;
    }
    xmin = i;
    validateParameters();
    xminChanged(xmin);
}

void GeneralizationWidget::updateXMax() {
    bool ok;
    int i = xminLineEdit->text().toInt(&ok);
    if (!ok) {
        i = 0;
    }
    xmax = i;
    validateParameters();
    xmaxChanged(xmax);
}

void GeneralizationWidget::updateYMin() {
    bool ok;
    int i = xminLineEdit->text().toInt(&ok);
    if (!ok) {
        i = 0;
    }
    ymin = i;
    validateParameters();
    yminChanged(ymin);
}

void GeneralizationWidget::updateYMax() {
    bool ok;
    int i = xminLineEdit->text().toInt(&ok);
    if (!ok) {
        i = 0;
    }
    ymax = i;
    validateParameters();
    ymaxChanged(ymax);
}

void GeneralizationWidget::updateZMin() {
    bool ok;
    int i = xminLineEdit->text().toInt(&ok);
    if (!ok) {
        i = 0;
    }
    zmin = i;
    validateParameters();
    zminChanged(zmin);
}

void GeneralizationWidget::updateZMax() {
    bool ok;
    int i = xminLineEdit->text().toInt(&ok);
    if (!ok) {
        i = 0;
    }
    zmax = i;
    validateParameters();
    zmaxChanged(zmax);
}*/

void GeneralizationSectionWidget::updateYStep(int i) {
    yStep = i;
    ystepChanged(yStep);
}

/*void GeneralizationWidget::updateData(QVector<QString> dataVector) {
    if (dataVector.size()!= seismicFiles.size()) {
        qDebug() << "GeneralizationWidget::updateData unexpected input : size do not match";
        return;
    }
    seismicFiles = dataVector;

    for ( int i=0; i<seismicFiles.size(); i++) {
        seismicEdit[i]->setText(seismicFiles[i]);
    }
}*/

void GeneralizationSectionWidget::updateScales(QVector<float> scalesVector) {
    if (scalesVector.size()!= scales.size()) {
        qDebug() << "GeneralizationWidget::updateData unexpected input : size do not match";
        return;
    }
    scales = scalesVector;

    for ( int i=0; i<scales.size(); i++) {
        scalesEdit[i]->setValue(scales[i]);
    }
}

/*void GeneralizationWidget::searchProgramFile() {
    QFileInfo file(program);

    QString val = QFileDialog::getOpenFileName(this, tr("Select program file"), file.absolutePath());
    if (!val.isEmpty() && !val.isNull()) {
        m_programEdit->setText(val);
    }
}

void GeneralizationWidget::searchWorkDir() {
    QFileInfo file(workDir);

    QString val = QFileDialog::getExistingDirectory(this, tr("Select config file"), file.absolutePath());
    if (!val.isEmpty() && !val.isNull()) {
        m_workDirEdit->setText(val);
    }
}

void GeneralizationWidget::searchCheckpoint() {
    QFileInfo file(checkpoint);

    QString val = QFileDialog::getOpenFileName(this, tr("Select config file"), file.absolutePath());
    if (!val.isEmpty() && !val.isNull()) {
        m_checkpointEdit->setText(val);
    }
}*/

void GeneralizationSectionWidget::setWellKind(QVector<QString> array) {
    logNames = array;

    setScaleUI();

    reset();
    /*
    // test if array is the same
    bool arrayIsSame = array.size() == logNames.size();
    int i=0;
    while (arrayIsSame && i<array.size()) {
        arrayIsSame = array[i].compare(logNames[i])==0;
        i++;
    }
    if (!arrayIsSame) {
        QFormLayout* layout = (QFormLayout*) m_scaleWidget->layout();

        while(array.size() != layout->count()) {
            if (array.size()<layout->count()) {

                layout->removeRow(layout->count()-1);
                scales.removeLast();
                scalesEdit.removeLast();
            } else if (array.size()>layout->count()) {
                QLineEdit* lineEdit = new QLineEdit;
                QDoubleValidator* validator = new QDoubleValidator(lineEdit);
                lineEdit->setValidator(validator);
                layout->addRow("", lineEdit);
                scalesEdit.append(lineEdit);
                scales.append(1.0);
                connect(lineEdit, &QLineEdit::textChanged, this, [this, lineEdit](QString s) {
                    int i=0;
                    while (i<scalesEdit.size() && lineEdit != scalesEdit[i]) {
                        i++;
                    }
                    if (i==scalesEdit.size()) {
                        return;
                    }
                    bool ok;
                    this->scales[i] = locale().toDouble(s, &ok);
                    this->validateParameters();
                });
                lineEdit->setText("1.0");
            }
        }

        for(int i=0; i<array.size(); i++) {
            QLabel* label = (QLabel*) layout->itemAt(i, QFormLayout::LabelRole)->widget();
            label->setText(array[i]);
        }

        validateParameters();
    }*/
}

void GeneralizationSectionWidget::updateDimensions() {
    if (seismicNames.size()>0) {
    	murat::io::InputOutputCube<float>* cube_seismic = murat::io::openCube<float>(seismicNames[0].second.toStdString());
        dimx = cube_seismic->getDim().getI();
        dimy = cube_seismic->getDim().getJ();
        dimz = cube_seismic->getDim().getK();
        seismicSampleRate = cube_seismic->getSteps().getI();
        stepy = cube_seismic->getSteps().getJ();
        stepz = cube_seismic->getSteps().getK();
        originy = cube_seismic->getOrigin().getJ();
        originz = cube_seismic->getOrigin().getK();
        delete cube_seismic;

        qDebug() << m_orientationComboBox->currentText();
        if (m_orientationComboBox->currentText().compare("inline")==0) {
            m_sliceSlider->setMaximum(originz+stepz*(dimz-1));
            m_sliceSlider->setMinimum(originz);
            m_sliceSlider->setSingleStep(stepz);
            m_sliceSpinBox->setMaximum(originz+stepz*(dimz-1));
            m_sliceSpinBox->setMinimum(originz);
            m_sliceSpinBox->setSingleStep(stepz);
            m_yStepSpinBox->setMaximum(dimy);
        } else {
            m_sliceSlider->setMaximum(originy+stepy*(dimy-1));
            m_sliceSlider->setMinimum(originy);
            m_sliceSlider->setSingleStep(stepy);
            m_sliceSpinBox->setMaximum(originy+stepy*(dimy-1));
            m_sliceSpinBox->setMinimum(originy);
            m_sliceSpinBox->setSingleStep(stepy);
            m_yStepSpinBox->setMaximum(dimz);
        }
    } else {
        dimx = 0;
        dimy = 0;
        dimz = 0;
        m_sliceSlider->setMaximum(0);
        m_sliceSlider->setMinimum(0);
        m_sliceSpinBox->setMaximum(0);
        m_sliceSpinBox->setMinimum(0);
    }
}

void GeneralizationSectionWidget::changeOrientation(QString txt) {
    if (dimx != 0) {
        if (txt.compare("inline")==0)  {
            orientation = SectionOrientation::INLINE;
            m_sliceSlider->setMaximum(originz+stepz*(dimz-1));
            m_sliceSlider->setMinimum(originz);
            m_sliceSlider->setSingleStep(stepz);
            m_sliceSpinBox->setMaximum(originz+stepz*(dimz-1));
            m_sliceSpinBox->setMinimum(originz);
            m_sliceSpinBox->setSingleStep(stepz);

            m_sliceSlider->show();
            m_sliceSpinBox->show();
            m_randomComboBox->hide();
            m_randomComboBox->clear();
        } else if(txt.compare("xline")==0) {
            orientation = SectionOrientation::XLINE;
            m_sliceSlider->setMaximum(originy+stepy*(dimy-1));
            m_sliceSlider->setMinimum(originy);
            m_sliceSlider->setSingleStep(stepy);
            m_sliceSpinBox->setMaximum(originy+stepy*(dimy-1));
            m_sliceSpinBox->setMinimum(originy);
            m_sliceSpinBox->setSingleStep(stepy);
            m_sliceSlider->show();
            m_sliceSpinBox->show();
            m_randomComboBox->hide();
            m_randomComboBox->clear();
        } else {
            orientation = SectionOrientation::RANDOM;
            m_sliceSlider->hide();
            m_sliceSpinBox->hide();
            m_randomComboBox->show();
            m_randomComboBox->addItem("");

            QDir randomDir(IJK_dir + "/RANDOMS");
            if (randomDir.exists()) {
                QFileInfoList randoms = randomDir.entryInfoList(QStringList()<< "*", QDir::Files, QDir::Name);
                for (QFileInfo& e : randoms) {
                    if (e.fileName().split(".").last().compare("xt")!=0) {
                        m_randomComboBox->addItem(e.baseName());
                    }
                }
            }
        }
        wellGraphicManager.setOrientation(orientation);
    }
    stack.clear();
}

void GeneralizationSectionWidget::setSeismicNames(QVector<std::pair<QString, QString>> array) {
    bool filesValid = true;
    QStringList invalidDatasets;
    for (int fIdx = 0; fIdx<array.size(); fIdx++) {
        QFileInfo fileInfo(array[fIdx].second);
        bool singleFileValid = fileInfo.exists() && fileInfo.isReadable();
        filesValid = filesValid && singleFileValid;

        if (!singleFileValid) {
            invalidDatasets.append(array[fIdx].first);
        }
    }

    if (filesValid) {
        seismicNames = array;
    } else {
        QMessageBox::critical(this, tr("Invalid data"), "The following files are invalid : "+invalidDatasets.join(","));
        seismicNames = QVector<std::pair<QString, QString>>();
    }
    updateDimensions();

    reset();
}
/*
    // test if array is the same
    bool arrayIsSame = array.size() == seismicNames.size();
    int i=0;
    while (arrayIsSame && i<array.size()) {
        arrayIsSame = array[i].compare(seismicNames[i])==0;
        i++;
    }
    if (!arrayIsSame) {
        clearSeismic();
        seismicNames = array;
        QFormLayout* layout = (QFormLayout*) m_dataWidget->layout();
        while(array.size() != layout->count()) {
            if (array.size()<layout->count()) {

                layout->removeRow(layout->count()-1);
                seismicEdit.removeLast();
                seismicButton.removeLast();
                seismicFiles.removeLast();
            } else if (array.size()>layout->count()) {
                QHBoxLayout* layout = new QHBoxLayout;
                QPushButton* loadFileButton = new QPushButton;
                QLineEdit* textField = new QLineEdit("");
                layout->addWidget(textField);
                layout->addWidget(loadFileButton);
                seismicButton.append(loadFileButton);
                seismicEdit.append(textField);
                connect(loadFileButton, &QPushButton::clicked, [this, textField]() {
                    QString filename = QFileDialog::getOpenFileName(this, tr("Open BBoxes file"), this->cacheDir, tr("Xt Files (*.xt)"));
                    if (!filename.isNull() && !filename.isEmpty()) {
                        this->cacheDir = QFileInfo(filename).absoluteDir().absolutePath();
                        textField->setText(filename);
                        textField->editingFinished();
                    }
                });

                seismicFiles.append("");
                connect(textField, &QLineEdit::textChanged, this, [this, textField](QString s) {
                    int i=0;
                    while (i<seismicFiles.size() && textField != seismicEdit[i]) {
                        i++;
                    }
                    if (i==seismicEdit.size()) {
                        return;
                    }
                    seismicFiles[i] = s;
                    validateParameters();
                });
                textField->setText("");
            }
        }

        for(int i=0; i<array.size(); i++) {
            QLabel* label = (QLabel*) layout->itemAt(i, QFormLayout::LabelRole)->widget();
            label->setText(array[i]);
        }

        validateParameters();
    }
}*/
void GeneralizationSectionWidget::generalizationFinished(QString checkpoint) {
    if (stack.size()==0) {
        qDebug() << "GeneralizationWidget::generalizationFinished Unexpected error";
        return;
    } else if (checkpoint.compare(*stack.begin())!=0) {
        predictFromStack();
        return;
    }
    stack.pop_front();

    QVector<Viewer>::iterator it = viewers.begin();
    int index = 0;
    while (it!=viewers.end() && (*it).file.compare(checkpoint)!=0) {
        it++;
        index ++;
    }

    if (it==viewers.end()) {
        qDebug() << "GeneralizationWidget::generalizationFinished Unexpected error";
        return;
    }



    int baseH = viewers[index].mat->height();
    int w = viewers[index].mat->width();

    QString name;
    if (m_orientationComboBox->currentText().compare("inline")==0 || m_orientationComboBox->currentText().compare("xline")==0) {
        name = survey + "/DATA/SEISMIC/";
    } else {
        name = IJK_dir + "/RANDOMS/";
    }
    name = name + "seismic3d." + logNames[selectedLogsIndexes[0]];
    if (!saveSuffix.isNull() && !saveSuffix.isEmpty()) {
        name += "_" + saveSuffix;
    }
    name += ".xt";
    viewers[index].mat->transpose();

    murat::io::InputOutputCube<float>* cube_seismic = murat::io::openCube<float>(name.toStdString());
    float cubeSampleRate = cube_seismic->getSteps().getI();
    int h = std::floor((baseH-1)*cubeSampleRate/currentPredictSampleRate)+1;
    std::vector<float> tmpBuf;
    tmpBuf.resize(w*cube_seismic->getDim().getI());
    if (m_orientationComboBox->currentText().compare("inline")==0) {
        cube_seismic->readSubVolume(0,0, sliderValueToVolumeIndex(), cube_seismic->getDim().getI(), cube_seismic->getDim().getJ(), 1, tmpBuf.data());
    } else if(m_orientationComboBox->currentText().compare("xline")==0) {
        cube_seismic->readSubVolume(0,sliderValueToVolumeIndex(), 0, cube_seismic->getDim().getI(), 1, cube_seismic->getDim().getK(), tmpBuf.data());
	} else {
        cube_seismic->readSubVolume(0, 0, 0, h, w, 1, tmpBuf.data());
    }

    if (cube_seismic->getDim().getI()==h) {
	   memcpy(viewers[index].mat->data(), tmpBuf.data(), tmpBuf.size()*sizeof(float));
   } else {
	   // if cube_seismic->getDim().getI() != h, it should be baseH
	   for (int traceIndex=0; traceIndex<w;traceIndex++) {
		   resampleSplineFloat(currentPredictSampleRate, cubeSampleRate, tmpBuf.data()+traceIndex*baseH, baseH, ((float*) viewers[index].mat->data())+traceIndex*h, h);
	   }
   }

    delete cube_seismic;
    viewers[index].mat->transpose();

    float min = viewers[index].paletteWidget->getRange().x();// * scales[0];
    float max = viewers[index].paletteWidget->getRange().y();// * scales[0];

    int N = viewers[index].paletteWidget->getLookupTable().size() - 1;
    if (N>255) {
        N = 255;
    }
    QVector<QRgb> colors;
    for (int c=0; c<=N; c++) {
        std::array<int, 4> array = viewers[index].paletteWidget->getLookupTable().getColors(c);
        colors.append(qRgba(array[0], array[1], array[2], array[3]));
    }
    QImage img(w, h, QImage::Format_Indexed8);
    img.setColorTable(colors);

    for (int j=0; j<h; j++) {
        for (int k=0; k<w; k++) {
            signed short val = N * (viewers[index].mat->getLine(j)[k] - min) / (max - min);
            if (val <0) {
                val = 0;
            } else if (val>N) {
                val = N;
            }
            img.scanLine(j)[k] = (unsigned char) val;
        }
    }
    viewers[index].view->setImage(img, (Matrix2DInterface*) viewers[index].mat);

//    for (int i=0; i<viewers.size(); i++) {
//        viewers[i].view->show();
//    }

    predictFromStack();
}

void GeneralizationSectionWidget::generalizationFailed(QString checkpoint) {
    if (stack.size()==0) {
        qDebug() << "GeneralizationWidget::generalizationFailed Unpexpected error";
    } else if (checkpoint.compare(*stack.begin())!=0) {
        predictFromStack();
    } else {
        // Do nothing because generalization seem to have been launched twice
        //stack.pop_front();
        //predictFromStack();
    }
}

void GeneralizationSectionWidget::generalizationRefused(QString checkpoint) {
    qDebug() << "GeneralizationWidget::generalizationRefused : mostlikely, the process already is already running";
}

void GeneralizationSectionWidget::setSurvey(QString txt) {
    survey = txt;
}

void GeneralizationSectionWidget::setYStep(int val) {
    m_yStepSpinBox->setValue(val);
    yStep = val;
}

void GeneralizationSectionWidget::setSuffix(QString txt) {
    m_suffixLineEdit->setText(txt);
}

void GeneralizationSectionWidget::setScales(QVector<float> array) {
    if (array.size()!=scalesEdit.size()) {
        qDebug() << "GeneralizationWidget::setScales : wrong input size";
        return;
    }

    scales = array;
    for (int i=0; i<scalesEdit.size(); i++) {
        scalesEdit[i]->setValue(array[i]);
    }
}

void GeneralizationSectionWidget::setPredictSampleRate(float predictSampleRate) {
	this->predictSampleRate = predictSampleRate;
    m_predictSampleRateSpinBox->setValue(predictSampleRate);
}

void GeneralizationSectionWidget::setPredictSampleRateInternal(double predictSampleRate) {
	this->predictSampleRate = predictSampleRate;

	emit predictSampleRateChanged(predictSampleRate);
}

void GeneralizationSectionWidget::setWells(QVector<Well> *wells) {
    this->wells = wells;
    m_jsonWellTreeWidget->clear();
    if (this->wells!=nullptr) {
        for (Well& well : (*wells)) {
            QTreeWidgetItem* item = new QTreeWidgetItem;
            item->setText(0, well.name);
            item->setData(0, Qt::CheckStateRole, QVariant::fromValue(false));
            m_jsonWellTreeWidget->addTopLevelItem(item);
        }
    }
}

void GeneralizationSectionWidget::setSeismicDynamic(QVector<std::pair<float, float>> array) {
    seismicDynamic = array;
}

void GeneralizationSectionWidget::setLogDynamic(QVector<std::pair<float, float>> array) {
    logDynamic = array;
    wellGraphicManager.setLogDynamic(logDynamic);
    qDebug() << "DEBUG setLogDynamic";
    for (auto& e : array) {
		qDebug() << e.first << e.second;
	}
}

void GeneralizationSectionWidget::setWorkDir(QString workDir) {
    if (!this->workDir.isNull() && !this->workDir.isEmpty()) {
        fsWatcher.removePath(this->workDir);
    }

    this->workDir = workDir;
    if (!this->workDir.isNull() && !this->workDir.isEmpty()) {
        fsWatcher.addPath(this->workDir);
    }

    updateCheckpointListWidget();
}

void GeneralizationSectionWidget::setNetwork(NeuralNetwork network) {
    if (this->network != network) {
        this->network = network;
        m_checkpointListWidget->clear();
        updateCheckpointListWidget();
    }
}

void GeneralizationSectionWidget::updateCheckpointListWidget() {
    if (workDir.isNull() || workDir.isEmpty()) {
        m_checkpointListWidget->clear();
    } else {
        QString extension;
        if (network==NeuralNetwork::Xgboost) {
            extension = "ubj";
        } else {
            extension = "index";
        }

        QDir dir(workDir);
        QFileInfoList files = dir.entryInfoList(QStringList() << "*."+extension, QDir::Files);
        QList<QListWidgetItem*> presentItems;
        for (QFileInfo& e : files) {
            QString item_name = e.baseName();
            QList<QListWidgetItem*> list = m_checkpointListWidget->findItems(item_name, Qt::MatchExactly);
            presentItems.append(list);
            if (list.size()==0 ) {
                bool test;
                int iteration_number;
                if (item_name.compare(CHECKPOINT_REFERENCE)==0) {
                    iteration_number = 0;
                    test = true;
                } else {
                    QStringList splittedName = item_name.split("-");
                    iteration_number = splittedName.last().toInt(&test);
                }

                if (test) {
                    IterationListWidgetItem* item = new IterationListWidgetItem(item_name);
                    item->setData(Qt::UserRole, e.dir().absoluteFilePath(e.completeBaseName()));
                    //item->setData(Qt::UserRole, e.absoluteFilePath());
                    item->setData(Qt::UserRole+1, iteration_number); // Needed to make the sorting work
                    m_checkpointListWidget->addItem(item);
                    presentItems.append(item);
                }
            } else if (list.size()>1) {
                qDebug() << "GeneralizationWidget::updateCheckpointListWidget Unexpected number of items coming from find";
            }
        }

        // TODO : Remove the items from file that no longer exist
    }
}

void GeneralizationSectionWidget::updateViewerListFromSelection() {
    QList<QListWidgetItem*> selection = m_checkpointListWidget->selectedItems();

    QList<Viewer> presentViewer;
    for (QListWidgetItem* item : selection) {
        QString file = item->data(Qt::UserRole).toString();
        bool test;
        int iter_num;
        if (item->text().compare(CHECKPOINT_REFERENCE)==0) {
            iter_num = 0;
            test = true;
        } else {
            QStringList splittedName = file.split("-");
            iter_num = splittedName.last().toInt(&test);
        }

        if (!test || iter_num<0) {
            continue;
        }

        QVector<Viewer>::Iterator it = viewers.begin();
        while (it!=viewers.end() && it->file.compare(file)!=0 ) {
            it++;
        }
        if (it == viewers.end()) {
            // Only add if it is a new selected item
            int h = std::floor((dimx-1)*seismicSampleRate/currentPredictSampleRate)+1;
            int w;
            if (m_orientationComboBox->currentText().compare("inline")==0) {
                w = dimy;
            } else if (m_orientationComboBox->currentText().compare("xline")==0) {
                w = dimz;
            } else {
                w = randomPts.count();
            }


            int N = viewers.size();
            //for (int i=0; i<logNames.size(); i++) {
            // create viewer
            Viewer viewer;
            viewer.view = new MuratCanvas2dFullView;
            viewer.view->initScene();
            viewer.view->toggleCurve(false);
            viewer.file = file;
            viewer.isSeismic = false;
            viewer.iter_number = iter_num;
            viewer.view->setWindowName(formatCheckpointName(viewer.file));

            //if (logPalette.count()==0) {
                PaletteWidget* paletteWidget = new PaletteWidget;
                PaletteHolder* paletteHolder = new PaletteHolder(logDynamic[selectedLogsIndexes[0]].first * scales[0], logDynamic[selectedLogsIndexes[0]].second * scales[0]);
                paletteWidget->setPaletteHolder(paletteHolder);
                paletteWidget->setLookupTable(ColorTableRegistry::DEFAULT());
                std::pair<PaletteWidget*, PaletteHolder*> pair(paletteWidget, paletteHolder);
                logPalette.append(pair);
                pair.first->hide();

                connect(paletteWidget, &PaletteWidget::lookupTableChanged, this, [this, paletteWidget]() {
                    this->updateViewerPalette(paletteWidget);
                });
                connect(paletteWidget, SELECT<float>::OVERLOAD_OF(&PaletteWidget::opacityChanged), this, [this, paletteWidget]() {
                    this->updateViewerPalette(paletteWidget);
                });
                connect(paletteWidget, &PaletteWidget::rangeChanged, this, [this, paletteWidget]() {
                    this->updateViewerPalette(paletteWidget);
                });
            //}
            viewer.paletteWidget = pair.first;
            /*connect(viewer.view , &MuratCanvas2dFullView::paletteRequestedSignal, this, [this](MuratCanvas2dFullView* view) {
                PaletteWidget* paletteWidget = nullptr;
                for (auto& e: viewers) {
                    if (paletteWidget!= e.paletteWidget && view != e.view) {
                        e.paletteWidget->hide();
                    } else if (view == e.view) {
                        paletteWidget = e.paletteWidget;
                        e.paletteWidget->show();
                    }
                }
            });*/
            connect(viewer.view , &MuratCanvas2dFullView::requestMenu, this, [this](QPoint pos, MuratCanvas2dFullView* view) {
                QMenu menu;
                //add default menu

                QAction* actionX = menu.addAction("Adjust palette");

                QObject::connect(actionX, &QAction::triggered, [this, view](bool checked) {
                    PaletteWidget* paletteWidget = nullptr;
                    for (auto& e: this->viewers) {
                        if (paletteWidget!= e.paletteWidget && view != e.view) {
                            e.paletteWidget->hide();
                        } else if (view == e.view) {
                            paletteWidget = e.paletteWidget;
                            e.paletteWidget->show();
                        }
                    }
                });

                int i=0;
                while (i<this->viewers.count() && this->viewers[i].view != view) {
                    i++;
                }
                if (i<this->viewers.count() && !this->viewers[i].isSeismic) {
                    QAction* actionY = menu.addAction("Save");
                    QObject::connect(actionY, &QAction::triggered, [this, view](bool checked) {
                        QString suffix = QInputDialog::getText(this, "Define suffix", "What suffix do you want to use ?", QLineEdit::Normal, m_suffixLineEdit->text()+"_save");
                        if (!suffix.isNull() && !suffix.isEmpty()) {
                            int i=0;
                            while (i<this->viewers.count() && this->viewers[i].view != view) {
                                i++;
                            }
                            QString checkpoint = this->viewers[i].file;

                            emit this->generalize(0, this->dimx-1, 0, this->dimy-1, 0, this->dimz-1, checkpoint, suffix);
                        }
                    });
                }

                QAction* actionY = menu.addAction("Adjust well plotting parameters");
                QObject::connect(actionY, &QAction::triggered, [this, view](bool checked) {
                	this->wellGraphicManager.runGraphicsSettingsDialog(this);
                });

                //show the menu
                menu.exec(pos);
            });



            //viewer.view->setWindowName(logNames[i]);


            QImage img(w, h, QImage::Format_Grayscale8);
            img.fill(0);
            if (viewer.mat) {
                delete viewer.mat;
            }
            viewer.mat = new Matrix2DLine<float>(w, h);
            memset(viewer.mat->data(), 0, w*h*sizeof(float));

            //viewer.view->hide();
            viewer.view->setImage(img, (Matrix2DInterface*) viewer.mat);

            QVector<Viewer>::const_iterator it = getFirstGUIValidCheckPointViewer();
            if (viewer.dockWidget==nullptr) {
                viewer.dockWidget = new KDDockWidgets::DockWidget(m_uniqueName+"_"+file.split("/").last());
                viewer.dockWidget->setWindowTitle(formatCheckpointName(file));
                viewer.dockWidget->setTitle(formatCheckpointName(file));
                viewer.dockWidget->setOptions(KDDockWidgets::DockWidget::Option_NotClosable);
                viewer.dockWidget->setWidget(viewer.view);
            }
            if (it==viewers.end()) {
                addDockWidget(viewer.dockWidget, KDDockWidgets::Location_OnRight, viewers[0].dockWidget);
            } else {
                it->dockWidget->addDockWidgetAsTab(viewer.dockWidget);
            }
            synchronizer.addCanvas2d(viewer.view);
            orderedInsertion(viewers, viewer);

            wellGraphicManager.connectViewer(viewer.view);
            stack.push_back(viewer.file);

            //viewer.view->show();
            //}

            presentViewer.append(viewer);
        } else {
            presentViewer.append(*it);
        }
    }

    for (int i=viewers.count()-1; i>=0; i--) {
        Viewer item = viewers[i];

        if (!item.isSeismic) {
            int it = 0;
            while(it<presentViewer.count() && presentViewer[it].view!=item.view) {
                it++;
            }
            if (it==presentViewer.count()) {
                QString name = item.file;

                stack.remove(name);

                viewers.remove(i);
                wellGraphicManager.disconnectViewer(item.view);
                synchronizer.remove(item.view);
//                gridLayout->removeWidget(item.view);
                item.dockWidget->deleteLater();
                item.view->deleteLater();
                delete item.mat;

                item.paletteWidget->deleteLater();

                auto it = std::find_if(logPalette.begin(), logPalette.end(), [item](const std::pair<PaletteWidget*, PaletteHolder*>& pair) {
                    return pair.first==item.paletteWidget;
                });
                if (it!=logPalette.end()) {
                    it->first->deleteLater();
                    delete it->second;
                    logPalette.erase(it);
                }
            }
        }
    }

    predictFromStack();
}

void GeneralizationSectionWidget::orderedInsertion(QVector<Viewer> &array, Viewer &item) {
    int N = array.size();
    /*int index = N/2;
    int down = 0;
    int up = N-1;

    while(index>=0 && index<N && array[index].iter_number!=item.iter_number) {
        if (item.iter_number < array[index].iter_number) {
            up = index - 1;
            index = (down + index)/2;
            if (index>up) {
                index = up;
            }

            if (up<down) {
                break;
            }
        } else {
            down = index + 1;
            index = (up + index)/2;
            if (index<down) {
                index = down;
            }

            if (up<down) {
                index = down;
                break;
            }
        }
    }
    array.insert(index, item);*/

    int i=0;
    while (i<N && item.iter_number>=array[i].iter_number) {
        i++;
    }
    array.insert(i, item);

}


void GeneralizationSectionWidget::predictFromStack() {
    if (stack.size()==0) {
        return;
    }

    QString checkpoint = *stack.begin();

    int slice = sliderValueToVolumeIndex();

    int ymin, ymax, zmin, zmax;
    if (m_orientationComboBox->currentText().compare("inline")==0) {
        ymin = 0;
        ymax = dimy-1;
        zmin = slice;
        zmax = slice;
        emit generalize(0, dimx-1, ymin, ymax, zmin, zmax, checkpoint, m_suffixLineEdit->text());
    } else if (m_orientationComboBox->currentText().compare("xline")==0) {
        ymin = slice;
        ymax = slice;
        zmin = 0;
        zmax = dimz-1;
        emit generalize(0, dimx-1, ymin, ymax, zmin, zmax, checkpoint, m_suffixLineEdit->text());
    } else {
        ymin = 0;
        ymax = randomPts.count()-1;
        zmin = 0;
        zmax = 0;

        QStringList data;
        QString random_filename =  IJK_dir+ "/RANDOMS/" + m_randomComboBox->currentText();
        for (std::pair<QString, QString> e : seismicNames) {
            data.append(random_filename+"_"+e.second.split("/").last().split(".")[1]+".xt");
        }
        emit generalizeRandom(0, dimx-1, ymin, ymax, zmin, zmax, checkpoint, m_suffixLineEdit->text(),
                              data, IJK_dir + "/RANDOMS");
        qDebug() << "GeneralizationWidget::launchGeneralization unexpected combobox input";
    }
}

void GeneralizationSectionWidget::updateViewerPalette(PaletteWidget* palette) {
    if (palette == nullptr) {
        return;
    }
    int index = 0;
    while (index < viewers.count()) {
        if(palette==viewers[index].paletteWidget) {
            int h = viewers[index].mat->height();
            int w = viewers[index].mat->width();
            QImage img(w, h, QImage::Format_Indexed8);
            float min = viewers[index].paletteWidget->getRange().x();// * scales[0];
            float max = viewers[index].paletteWidget->getRange().y();// * scales[0];
            //QImage img = viewers[index].view->getQImage();

            int N = viewers[index].paletteWidget->getLookupTable().size() - 1;
            if (N>255) {
                N = 255;
            }
            QVector<QRgb> colors;
            for (int c=0; c<=N; c++) {
                std::array<int, 4> array = viewers[index].paletteWidget->getLookupTable().getColors(c);
                colors.append(qRgba(array[0], array[1], array[2], array[3]));
            }
            img.setColorTable(colors);

            for (int j=0; j<h; j++) {
                for (int k=0; k<w; k++) {
                    signed short val = N * (viewers[index].mat->getLine(j)[k] - min) / (max - min);
                    if (val <0) {
                        val = 0;
                    } else if (val>N) {
                        val = N;
                    }
                    img.scanLine(j)[k] = (unsigned char) val;
                }
            }
            viewers[index].view->setImage(img, (Matrix2DInterface*) viewers[index].mat);
        }
        index++;
    }
}

/*void GeneralizationSectionWidget::findIJKDirectory() {
	IJK_dir = "";

	bool goal_reached = false;
	int i = 0;
	int N = seismicNames.size()+1;

	QString basePath = survey + "/ImportExport/IJK/";
	while (!goal_reached && i<N) {
		QString addOnPath = "";
		if (i>0) {
			// addOnPath use seismic name
			QString seismicFilePath = seismicNames[i-1].second;
			QString descFilePath = seismicFilePath.section(".", 0, -2) + ".desc";
			QFileInfo descFileInfo = QFileInfo(descFilePath);
			if (descFileInfo.exists() && descFileInfo.isFile()) { // if desc is here we use it to find sismage name
				QFile descFile(descFilePath);
				if (descFile.open(QIODevice::ReadOnly)) {
					QTextStream stream(&descFile);
					QString line;
					bool test = false;

					while (!test && stream.readLineInto(&line)) {
						//qDebug() << "DEBUG line : " << line << "-section-" << line.section("=", 0, 0) << (line.section("=", 0, 0).compare("name")==0) << (line.section("=", 0, 0).compare(QString("name"))==0);
						test = line.section("=", 0, 0).compare("name")==0;
						if(test) {
							addOnPath = line.section("=", 1);
						}
					}
				}
			} else { // else we use the name from seismic filename
				addOnPath = seismicFilePath.section("/", -1).section(".", 1, 1);
			}
		} // else i=0 => no addOnPath

		QString path = basePath + addOnPath + "/WELL_TRAJECTORIES";

		goal_reached = QDir(path).exists();
		if (goal_reached) {
			IJK_dir = basePath + addOnPath + "/";
		}

		i++;
	}
}*/

void GeneralizationSectionWidget::setIJKDirectory(QString ijk_dir) {
	IJK_dir = ijk_dir;


    m_ijkWellListWidget->clear();
    m_ijkWellListWidget->addItem("");
    QDir dir(IJK_dir+"/WELL_TRAJECTORIES");
    if (dir.exists()) {
        QFileInfoList infoList = dir.entryInfoList(QStringList() << "*.txt", QDir::Files);
        for (QFileInfo& info : infoList) {
            trajectories.append(info.baseName());
            m_ijkWellListWidget->addItem(info.baseName());
        }
    } else {
    	qDebug() << "IJK Dir missing : " << IJK_dir+"/WELL_TRAJECTORIES";
        m_ijkWellListWidget->clear();
        QMessageBox::warning(this, tr("Well trajectories missing"), tr("Please, run the IJK program."));
    }
}

void GeneralizationSectionWidget::setSelectedLogsIndexes(QVector<int> logsIndexes) {
	selectedLogsIndexes = logsIndexes;
	wellGraphicManager.setValueIndex(logsIndexes[0]);
	setScaleUI();
}

void GeneralizationSectionWidget::setScaleUI() {
	if (selectedLogsIndexes.size()<=0 || logNames.size()<selectedLogsIndexes.size() || logNames.size()<=0) {
		return;
	}

    scales.clear();
    for (auto e : selectedLogsIndexes) {
        scales.append(1);
    }
    emit scalesChanged(scales);


    for (int i=scalesEdit.size()-1; i>=0; i--) {
        QDoubleSpinBox* e = scalesEdit[i];
        scalesEdit.pop_back();

        m_scaleWidgetLayout->removeRow(i); // the widgets in the row are deleted
    }

    for (int i=0; i<scales.size(); i++) {
        QDoubleSpinBox* e = new QDoubleSpinBox;
        e->setMinimum(std::numeric_limits<float>::min());
        e->setMaximum(std::numeric_limits<float>::max());
        e->setValue(scales[i]);
        scalesEdit.append(e);
        m_scaleWidgetLayout->addRow(logNames[selectedLogsIndexes[i]], e);

        connect(e, SELECT<double>::OVERLOAD_OF(&QDoubleSpinBox::valueChanged), this, [i, this](double val) {
            scales[i] = val;
            emit scalesChanged(scales);
        });
    }

    reset();
}

void GeneralizationSectionWidget::setTrainSampleRate(float trainSampleRate) {
	this->trainSampleRate = trainSampleRate;

	reset();
}

int GeneralizationSectionWidget::sliderValueToVolumeIndex() {
    int volumeIndex = 0;
    if (m_orientationComboBox->currentText().compare("inline")==0) {
        volumeIndex = (m_sliceSlider->value() - originz) / stepz;
        if (volumeIndex>=dimz) {
            volumeIndex = dimz-1;
        }
    } else if (m_orientationComboBox->currentText().compare("xline")==0) {
        volumeIndex = (m_sliceSlider->value() - originy) / stepy;
        if (volumeIndex>=dimy) {
            volumeIndex = dimy-1;
        }
    }
    if (volumeIndex<0) {
        volumeIndex = 0;
    }
    return volumeIndex;
}

void GeneralizationSectionWidget::jsonWellSelectionChanged() {
	QList<QTreeWidgetItem*> items = m_jsonWellTreeWidget->selectedItems();
	QList<QTreeWidgetItem*> newSelection;
	std::vector<bool> isItemInSelection;
	isItemInSelection.resize(loadedJsonWells.size(), false);

	for (QTreeWidgetItem* item : items) {
		std::size_t index_ref = 0;
		while(index_ref<loadedJsonWells.size() && item->text(0).compare(loadedJsonWells[index_ref])) {
			index_ref++;
		}
		if (index_ref==loadedJsonWells.size()) {
			newSelection.push_back(item);
		} else {
			isItemInSelection[index_ref] = true;
		}
	}

	for(long index_ref=loadedJsonWells.size()-1; index_ref>=0; index_ref--) {
		if (!isItemInSelection[index_ref]) {
			unloadJsonWell(loadedJsonWells[index_ref]);
		}
	}

	for (QTreeWidgetItem* item : newSelection) {
		loadJsonWell(item->text(0)); // use column 0 by default
	}
}

void GeneralizationSectionWidget::loadJsonWell(const QString& wellName) {
    int index = 0;
    while (index<(*wells).size() && (*wells)[index].name.compare(wellName)!=0) {
        index ++;
    }
    if (index < (*wells).size()) {
        Well well = (*wells)[index];
        QVector<QVector3D> jsonTrajectory;
        for (LogSample& logSample : well.samples) {
            jsonTrajectory.append(QVector3D(logSample.x, logSample.y, logSample.z));
        }

        this->wellGraphicManager.addWell(well, jsonTrajectory);
        loadedJsonWells.append(wellName);
    }
}

void GeneralizationSectionWidget::unloadJsonWell(const QString& wellName) {
    wellGraphicManager.removeWell(wellName);
    loadedJsonWells.removeOne(wellName);
}

void GeneralizationSectionWidget::jsonWellModelDataChanged(const QModelIndex& topLeft,
        const QModelIndex& bottomRight, const QVector<int>& roles) {
    for (int i : roles) {
        if (i == Qt::CheckStateRole) {
            QVariant var = topLeft.data(Qt::DisplayRole);
            QString wellName = var.toString();

            if (topLeft.data(Qt::CheckStateRole).toBool()) {
                QModelIndex parentModelIndex = topLeft.parent();
                QAbstractItemModel* model = m_jsonWellTreeWidget->model();
                QItemSelectionModel* selectionModel = m_jsonWellTreeWidget->selectionModel();
                selectionModel->select(topLeft, QItemSelectionModel::Select);
                for (int i=0; i<model->rowCount(parentModelIndex); i++) {
                    QModelIndex sibling = topLeft.sibling(i, topLeft.column());
                    if (sibling.isValid() && sibling.row()!=topLeft.row()) {
                        model->setData(sibling, QVariant::fromValue(false), Qt::CheckStateRole);
                    }
                }

                int index = 0;
                while (index<(*wells).size() && (*wells)[index].name.compare(wellName)!=0) {
                    index ++;
                }
                if (index < (*wells).size() && (*wells)[index].samples.size()>0) {
                    int newVal = 0;
                    bool changeValue = false;
                    if (m_orientationComboBox->currentText().compare("inline")==0) {
                        newVal = (*wells)[index].samples[0].x*stepz+originz;
                        changeValue = true;
                    } else if (m_orientationComboBox->currentText().compare("xline")==0) {
                        newVal = (*wells)[index].samples[0].y*stepy+originy;
                        changeValue = true;
                    }

                    changeValue = changeValue && newVal>=m_sliceSlider->minimum() && newVal<m_sliceSlider->maximum() &&
                            newVal!=m_sliceSlider->value();

                    if (changeValue) {
                        m_sliceSlider->setValue(newVal);
                        m_sliceSlider->setSliderDown(true);
                        m_sliceSlider->setSliderDown(false);
                    }
                }
            }
        }
    }
}

QVector<GeneralizationSectionWidget::Viewer>::const_iterator GeneralizationSectionWidget::getFirstGUIValidCheckPointViewer() const {
	QVector<Viewer>::const_iterator it = std::find_if(viewers.begin(), viewers.end(), [](const Viewer& viewer) {
		return !viewer.isSeismic && viewer.dockWidget!=nullptr;
	});
	return it;
}

QString GeneralizationSectionWidget::formatCheckpointName(const QString& filePath) {
	QString name = filePath.split("/").last();
	QString newName = name.replace("ckpt", "ckp").replace("_blind", "_b");
	return newName;
}

void GeneralizationSectionWidget::setupGui() {
    m_paramsDockWidget = new KDDockWidgets::DockWidget(m_uniqueName+"_paramsDockWidget");
    m_paramsDockWidget->setWindowTitle("Params");
    m_paramsDockWidget->setTitle("Params");
    m_paramsDockWidget->setWindowFlag(Qt::SubWindow,true);
    m_paramsDockWidget->setOptions(KDDockWidgets::DockWidget::Option_NotClosable);
    addDockWidget(m_paramsDockWidget, KDDockWidgets::Location_OnTop);

    QWidget* paramsHolder = new QWidget;
    QSizePolicy groupBoxPolicy = paramsHolder->sizePolicy();
    groupBoxPolicy.setVerticalPolicy(QSizePolicy::Policy::Fixed);
    paramsHolder->setSizePolicy(groupBoxPolicy);
    QVBoxLayout* paramsLayout = new QVBoxLayout;
    paramsHolder->setLayout(paramsLayout);
    m_paramsDockWidget->setWidget(paramsHolder);

    QHBoxLayout* formLikeLayout = new QHBoxLayout;
    paramsLayout->addLayout(formLikeLayout);
    formLikeLayout->addWidget(new QLabel("Save suffix"));
    m_suffixLineEdit = new QLineEdit("tmp_");
    formLikeLayout->addWidget(m_suffixLineEdit);
    formLikeLayout->addWidget(new QLabel("Sample rate"));
    m_predictSampleRateSpinBox = new QDoubleSpinBox;
    m_predictSampleRateSpinBox->setValue(1.0);
    formLikeLayout->addWidget(m_predictSampleRateSpinBox);
    formLikeLayout->addWidget(new QLabel("Y Step"));
    m_yStepSpinBox = new QSpinBox;
    m_yStepSpinBox->setMinimum(1);
    m_yStepSpinBox->setMaximum(99999);
    m_yStepSpinBox->setValue(100);
    formLikeLayout->addWidget(m_yStepSpinBox);

    m_scaleWidget = new QGroupBox("Scales");
    m_scaleWidgetLayout = new QFormLayout;
    m_scaleWidget->setLayout(m_scaleWidgetLayout);
    paramsLayout->addWidget(m_scaleWidget);

    QHBoxLayout* slicingLayout = new QHBoxLayout;
    paramsLayout->addLayout(slicingLayout);
    m_orientationComboBox = new QComboBox;
    m_orientationComboBox->addItem("inline");
    m_orientationComboBox->addItem("xline");
    m_orientationComboBox->addItem("random");
    slicingLayout->addWidget(m_orientationComboBox);
    m_sliceSlider = new QSlider(Qt::Horizontal);
    slicingLayout->addWidget(m_sliceSlider);
    m_sliceSpinBox = new QSpinBox;
    slicingLayout->addWidget(m_sliceSpinBox);
    m_randomComboBox = new QComboBox;
    slicingLayout->addWidget(m_randomComboBox);

    m_runDockWidget = new KDDockWidgets::DockWidget(m_uniqueName+"_runDockWidget");
    m_runDockWidget->setWindowTitle("Run button");
    m_runDockWidget->setTitle("Run button");
    m_runDockWidget->setWindowFlag(Qt::SubWindow,true);
    m_runDockWidget->setOptions(KDDockWidgets::DockWidget::Option_NotClosable);
    m_updateButton = new QPushButton("Update");
    m_runDockWidget->setWidget(m_updateButton);
    addDockWidget(m_runDockWidget, KDDockWidgets::Location_OnBottom);

    m_checkPointsDockWidget = new KDDockWidgets::DockWidget(m_uniqueName+"_checkPointsDockWidget");
    m_checkPointsDockWidget ->setWindowTitle("Checkpoints");
    m_checkPointsDockWidget->setTitle("Checkpoints");
    m_checkPointsDockWidget->setWindowFlag(Qt::SubWindow,true);
    m_checkPointsDockWidget->setOptions(KDDockWidgets::DockWidget::Option_NotClosable);
    m_checkpointListWidget = new QListWidget;
    m_checkpointListWidget->setSelectionMode(QAbstractItemView::MultiSelection);
    m_checkPointsDockWidget->setWidget(m_checkpointListWidget);
    addDockWidget(m_checkPointsDockWidget, KDDockWidgets::Location_OnLeft);

    m_ijkWellDockWidget = new KDDockWidgets::DockWidget(m_uniqueName+"_ijkWellsDockWidget");
    m_ijkWellDockWidget->setWindowTitle("IJK Wells");
    m_ijkWellDockWidget->setTitle("IJK Wells");
    m_ijkWellDockWidget->setWindowFlag(Qt::SubWindow,true);
    m_ijkWellDockWidget->setOptions(KDDockWidgets::DockWidget::Option_NotClosable);
    m_ijkWellListWidget = new QListWidget;
    m_ijkWellListWidget->setSelectionMode(QAbstractItemView::MultiSelection);
    m_ijkWellDockWidget->setWidget(m_ijkWellListWidget);
    addDockWidget(m_ijkWellDockWidget, KDDockWidgets::Location_OnBottom, m_checkPointsDockWidget);

    m_jsonWellDockWidget = new KDDockWidgets::DockWidget(m_uniqueName+"_jsonWellsDockWidget");
    m_jsonWellDockWidget->setWindowTitle("Json Wells");
    m_jsonWellDockWidget->setTitle("Json Wells");
    m_jsonWellDockWidget->setWindowFlag(Qt::SubWindow,true);
    m_jsonWellDockWidget->setOptions(KDDockWidgets::DockWidget::Option_NotClosable);
    m_jsonWellTreeWidget = new QTreeWidget;
    m_jsonWellTreeWidget->setSelectionMode(QAbstractItemView::MultiSelection);
    m_jsonWellTreeWidget->setHeaderHidden(true);
    m_jsonWellDockWidget->setWidget(m_jsonWellTreeWidget);
    addDockWidget(m_jsonWellDockWidget, KDDockWidgets::Location_OnBottom, m_ijkWellDockWidget);
}
