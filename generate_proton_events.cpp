#include "Pythia8/Pythia.h"
#include <iostream>
#include <cmath>
#include <random>

using namespace Pythia8;

int main(int argc, char* argv[]) {
  
  if (argc != 4) {
      std::cerr << "Usage: " << argv[0] << " <no of output events>  <output filename>  <energy of the proton beam>" << std::endl;
      return 1;
  }

  int nprotonEvents = std::stoi(argv[1]);
  std::string baseFilename = argv[2];
  std::string Eproton_beam = argv[3];
  std::string outputFileName = "pythia8_events/proton_events_" + baseFilename + ".txt";

  // Create and open a file for output
  std::ofstream outFile(outputFileName);
  if (!outFile.is_open()) {
      std::cerr << "Error: Could not open file for writing." << std::endl;
      return 1;
  }

  // Add header to the output file
  outFile << "# event_number particle_count E px py pz" << std::endl;

  // Random number generator
  std::random_device rd; // Used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Mersenne Twister engine, a good choice for most cases
  std::uniform_real_distribution<double> unif(0.0, 1.0);


  // Generator
  Pythia pythia;

  // Setting up the Monash tune for p-p collisions
  pythia.readString("Tune:pp = 14");


  // Set up proton on proton at 120 GeV per beam (15.5 GeV collision energy)
  // pythia.readString("Beams:eCM = 15.5");  // Center of mass energy in GeV
  // pythia.readString("Beams:eCM = "+Eproton_beam);  // Center of mass energy in GeV
  
  
  // Beam settings for one beam at rest and the other at 120 GeV
  // pythia.readString("Beams:idA = 2212"); // proton
  // pythia.readString("Beams:idB = 2212"); // proton
  pythia.readString("Beams:eA = "+Eproton_beam); // proton beam energy
  pythia.readString("Beams:eB = 0.0");   // proton at rest
  pythia.readString("Beams:frameType = 2"); // fixed target

  // Set the minimum pT for hard QCD (perturbative) processes (formally divergent...)
  pythia.readString("PhaseSpace:pTHatMin = 0.01");


  // Enable parton-level processes (ISR, FSR, MPI, beam remnants).
  pythia.readString("PartonLevel:ISR = on"); 
  pythia.readString("PartonLevel:FSR = on"); 
  pythia.readString("PartonLevel:MPI = on"); 
  // The Remnants switch is on by default, but you can be explicit:
  pythia.readString("PartonLevel:Remnants = on"); 

  // Enable hadronization.
  pythia.readString("HadronLevel:all = on");

  // Allow decays (including tau decays) to occur
  pythia.readString("ProcessLevel:all = on");
  pythia.readString("HadronLevel:Decay = on");
  
  // // Enable QCD processes which are necessary for hadron production
  // pythia.readString("HardQCD:all = on");
  pythia.readString("HardQCD:hardccbar = on");
  pythia.readString("HardQCD:hardbbbar = on");

  // // Allow soft QCD, which is more appropriate at low energies
  pythia.readString("SoftQCD:all = on");
  pythia.readString("Onia:all = on");
  pythia.readString("SoftQCD:nonDiffractive = on");
  pythia.readString("SoftQCD:singleDiffractive = on");
  pythia.readString("SoftQCD:doubleDiffractive = on");
  pythia.readString("SoftQCD:centralDiffractive = on");

  // Elastic
  pythia.readString("SoftQCD:elastic = on");

  // Decrease verbosity
  pythia.readString("Init:showProcesses = off");   // Suppress process listing at initialization
  pythia.readString("Init:showMultipartonInteractions = off");
  pythia.readString("Init:showChangedSettings = off");
  pythia.readString("Init:showAllSettings = off");
  pythia.readString("Init:showChangedParticleData = off");
  pythia.readString("Next:numberCount = 0");       // Disable periodic event listing
  pythia.readString("Next:numberShowInfo = 0");
  pythia.readString("Next:numberShowProcess = 0");
  pythia.readString("Next:numberShowEvent = 0");

  // Set error and warning verbosity
  pythia.readString("Print:quiet = on");           // Reduce printing of warnings
  pythia.settings.flag("Check:event", false);      // Disable extended event checking


  // Initialize the generator
  pythia.init();

  // Define counter for tau particles
  int protonCount = 0;
  int eventNumber = 0;

  while (protonCount < nprotonEvents) {
    if (!pythia.next()) continue;
    
    // Generated a new event
    ++eventNumber;
  
    for (int i = 0; i < pythia.event.size(); ++i) {
    
      int id = pythia.event[i].id();
    
      // Final state proton found
      if (id == 2212 && pythia.event[i].isFinal()) {

          ++protonCount;
          // Write particle data to the file with double precision
          outFile << std::fixed << std::setprecision(10)
          << eventNumber << " "
          << protonCount << " " 
          << pythia.event[i].e() << " " 
          << pythia.event[i].px() << " " 
          << pythia.event[i].py() << " " 
          << pythia.event[i].pz() << " "
          << std::endl;
          
        }

        // // D+/D- meson found (skipping D* decay for now)
        // if (abs(id) == 411 || abs(id) == 413) {
        //   ++DCount;

        //   outFile << generatedTauEvents << " " 
        //         << id << " " 
        //         << pythia.event[i].status() << " " 
        //         << pythia.event[i].mother1() << " " 
        //         << pythia.event[i].mother2() << " " 
        //         << pythia.event[i].daughter1() << " " 
        //         << pythia.event[i].daughter2() << " " 
        //         << pythia.event[i].e() << " " 
        //         << pythia.event[i].px() << " " 
        //         << pythia.event[i].py() << " " 
        //         << pythia.event[i].pz() << " "
        //         << 1 << std::endl; 
     

        //   // // Get the D meson's four-momentum
        //   Vec4 pD = pythia.event[i].p();

        //   // // Perform a toy model decay D -> nu_tau + tau
        //   double beta = pD.pAbs() / pD.e();
        //   double gamma = pD.e() / mD;

        //   // // // Assume isotropic decay for simplicity
        //   double phi = 2 * M_PI * unif(gen);
        //   double theta = acos(1 - 2 * unif(gen));

        //   // Tau and nu_tau momenta in D rest frame
        //   double pMagRest = sqrt((mD - mTau) * (mD + mTau));
        //   Vec4 pTauRest(pMagRest * sin(theta) * cos(phi),
        //                 pMagRest * sin(theta) * sin(phi),
        //                 pMagRest * cos(theta),
        //                 mTau);
        
        //   // 2 body decay kinematics
        //   Vec4 pNuTauRest = Vec4(0, 0, 0, 0) - pTauRest;

        //   // Boost to this event's frame
        //   pTauRest.bst(pD / pD.mCalc());
        //   pNuTauRest.bst(pD / pD.mCalc());

        //   // Write particle data to the file
        //   outFile << generatedTauEvents << " " 
        //         << ((id > 0) ? 15 : ((id < 0) ? -15 : 0)) << " " 
        //         << pythia.event[i].status() << " " 
        //         << pythia.event[i].mother1() << " " 
        //         << pythia.event[i].mother2() << " " 
        //         << pythia.event[i].daughter1() << " " 
        //         << pythia.event[i].daughter2() << " " 
        //         << pTauRest.e() << " " 
        //         << pTauRest.px() << " " 
        //         << pTauRest.py() << " " 
        //         << pTauRest.pz() << " "
        //         << 1.2e-3 << std::endl; // Br(D -- > nutau tau)
        // }

        // // Ds+/Ds- meson found
        // if (abs(id) == 431 || abs(id) == 433) {
        //   ++DCount;

        //   outFile << generatedTauEvents << " " 
        //         << id << " " 
        //         << pythia.event[i].status() << " " 
        //         << pythia.event[i].mother1() << " " 
        //         << pythia.event[i].mother2() << " " 
        //         << pythia.event[i].daughter1() << " " 
        //         << pythia.event[i].daughter2() << " " 
        //         << pythia.event[i].e() << " " 
        //         << pythia.event[i].px() << " " 
        //         << pythia.event[i].py() << " " 
        //         << pythia.event[i].pz() << " "
        //         << 1 << std::endl; 
     

        //   // // Get the D meson's four-momentum
        //   Vec4 pD = pythia.event[i].p();

        //   // // Perform a toy model decay D -> nu_tau + tau
        //   double beta = pD.pAbs() / pD.e();
        //   double gamma = pD.e() / mDs;

        //   // // // Assume isotropic decay for simplicity
        //   double phi = 2 * M_PI * unif(gen);
        //   double theta = acos(1 - 2 * unif(gen));

        //   // Tau and nu_tau momenta in D rest frame
        //   double pMagRest = sqrt((mDs - mTau) * (mDs + mTau));
        //   Vec4 pTauRest(pMagRest * sin(theta) * cos(phi),
        //                 pMagRest * sin(theta) * sin(phi),
        //                 pMagRest * cos(theta),
        //                 mTau);
        
        //   // 2 body decay kinematics
        //   Vec4 pNuTauRest = Vec4(0, 0, 0, 0) - pTauRest;

        //   // Boost to this event's frame
        //   pTauRest.bst(pD / pD.mCalc());
        //   pNuTauRest.bst(pD / pD.mCalc());
          
        //   // Write particle data to the file
        //   outFile << generatedTauEvents << " " 
        //         << ((id > 0) ? 15 : ((id < 0) ? -15 : 0)) << " " 
        //         << pythia.event[i].status() << " " 
        //         << pythia.event[i].mother1() << " " 
        //         << pythia.event[i].mother2() << " " 
        //         << pythia.event[i].daughter1() << " " 
        //         << pythia.event[i].daughter2() << " " 
        //         << pTauRest.e() << " " 
        //         << pTauRest.px() << " " 
        //         << pTauRest.py() << " " 
        //         << pTauRest.pz() << " "
        //         << 5.36e-2 << std::endl; // Br(Ds -- > nutau tau)

        // }

      }
    // }
    // if (hasTauOrDmeson) {generatedTauEvents++;}

  }
  // Close the output file
  outFile.close();

  return 0;
}
