import React, {
  useEffect,
  useMemo,
  useState,
} from "react";
import {
  ActivityIndicator,
  Alert,
  Image,
  Platform,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";

import AsyncStorage from "@react-native-async-storage/async-storage";
import * as FileSystem from "expo-file-system";

import * as ImagePicker from "expo-image-picker";
import * as DocumentPicker from "expo-document-picker";
import { useVideoPlayer, VideoView } from "expo-video";

const BACKEND_URL =
  process.env.EXPO_PUBLIC_BACKEND_URL ||
  "http://formcheck-ai-api-v3.eba-pvfk7qtv.us-west-2.elasticbeanstalk.com";

const API_URL = BACKEND_URL;

const MEDIA_URL = BACKEND_URL;

console.log("CURRENT API URL:", API_URL);
console.log("CURRENT MEDIA URL:", MEDIA_URL);

const fullUrl = (path) => {
  if (!path) return null;
  if (String(path).startsWith("http")) return path;
  return `${MEDIA_URL}${path}`;
};

const isPhaseImageUrl = (value) =>
  typeof value === "string" &&
  (
    value.startsWith("/") ||
    value.startsWith("http://") ||
    value.startsWith("https://")
  );

const hasRealPhaseImages = (phaseImages) => {
  if (!phaseImages) return false;
  return Object.values(phaseImages).some(
    (v) => typeof v === "string" && v.startsWith("/")
  );
};

const stripFrameNumberPhaseImages = (data) => {
  if (
    data?.phase_images &&
    !hasRealPhaseImages(data.phase_images)
  ) {
    return { ...data, phase_images: null };
  }
  return data;
};


const formatLabel = (v) =>
  v
    ? String(v)
        .replaceAll("_", " ")
        .replace(/\b\w/g, (c) => c.toUpperCase())
    : "N/A";

const getStatusColor = (status) => {
  const value = String(status || "good").toLowerCase();

  if (
    [
      "poor",
      "incomplete",
      "drifting",
      "severe_flare",
      "limited_range",
      "shallow",
      "high",
      "soft",
      "weak",
      "slow",
      "sagging",
      "knee_cave",
      "leaning",
      "leg_drive",
      "excessive",
      "stiff",
      "short",
      "severe",
      "off",
      "disconnected",
      "early_press",
      "bent",
    ].includes(value)
  ) {
    return "#ef4444";
  }

  if (
    [
      "borderline",
      "needs_work",
      "possible",
      "fair",
      "possibly_shallow",
      "review",
      "minor_knee_bend",
      "leaning_back",
      "leaning_forward",
      "unknown",
    ].includes(value)
  ) {
    return "#f59e0b";
  }

  return "#22c55e";
};

const getScoreColor = (score) => {
  if (score >= 9) return "#22c55e";
  if (score >= 7.5) return "#84cc16";
  if (score >= 6) return "#f59e0b";
  return "#ef4444";
};

const getBestRep = (reps) => {
  if (!reps || reps.length === 0) return null;

  return reps.reduce((best, rep) => {
    return Number(rep.score || 0) > Number(best.score || 0) ? rep : best;
  }, reps[0]);
};

const pickBreakdown = (breakdown, ...keys) => {
  for (const key of keys) {
    const value = breakdown?.[key];
    if (value !== undefined && value !== null && String(value).trim() !== "") {
      return value;
    }
  }
  return "good";
};

const findCoachingSection = (coaching, ...titles) => {
  const sections = coaching?.sections || [];
  for (const title of titles) {
    const match = sections.find(
      (s) =>
        String(s.title || "").toLowerCase() === String(title).toLowerCase()
    );
    if (match) return match;
  }
  return null;
};

const coachingSectionStatus = (section) => {
  if (!section) return null;
  if (section.status === "poor") return "poor";
  if (section.status === "warning") return "borderline";
  return "good";
};

const resolveZoneStatus = ({
  breakdown = {},
  coaching,
  breakdownKeys = [],
  coachingTitles = [],
}) => {
  const fromCoaching = coachingSectionStatus(
    findCoachingSection(coaching, ...coachingTitles)
  );
  if (fromCoaching) return fromCoaching;
  return pickBreakdown(breakdown, ...breakdownKeys);
};

const resolveZoneNote = ({
  coaching,
  coachingTitles = [],
  fallback = "",
  bestRep,
}) => {
  const section = findCoachingSection(coaching, ...coachingTitles);
  if (section?.message) return section.message;
  if (bestRep?.feedback?.length) return bestRep.feedback[0];
  return fallback;
};

const resolvePhaseImagePath = (phaseImages, key) => {
  if (!phaseImages) return null;

  const aliases = {
    setup: ["setup"],
    first_pull: ["first_pull"],
    transition: ["transition", "first_pull", "extension"],
    power_position: ["power_position", "extension"],
    extension_v2: ["extension_v2", "true_extension", "extension"],
    pull_under: ["pull_under", "extension"],
    catch: ["catch", "clean_catch"],
    recovery: ["recovery", "finish", "lockout"],
  };

  const candidates = aliases[key] || [key];

  for (const candidate of candidates) {
    if (phaseImages[candidate]) {
      return phaseImages[candidate];
    }
  }

  return null;
};

const alignZonesToPhaseReview = (
  zones,
  phaseConfig,
  phaseImages
) => {
  if (!Array.isArray(zones) || !phaseConfig?.items?.length) {
    return zones || [];
  }

  const phaseEntries = phaseConfig.items.map(([key, label]) => ({
    key,
    label,
    path: resolvePhaseImagePath(phaseImages, key),
  }));

  return zones.map((zone) => {
    const zonePath = resolvePhaseImagePath(
      phaseImages,
      zone?.imageKey
    );

    if (!zonePath) {
      return zone;
    }

    const matchingPhase = phaseEntries.find(
      (phase) => phase.path && phase.path === zonePath
    );

    if (!matchingPhase) {
      return zone;
    }

    return {
      ...zone,
      imageKey: matchingPhase.key,
      phaseLabel: matchingPhase.label,
    };
  });
};

const buildPhaseMatchedZones = (
  coachingZones,
  phaseConfig,
  phaseImages
) => {
  if (!phaseConfig?.items?.length) {
    return coachingZones || [];
  }

  const sourceZones = Array.isArray(coachingZones)
    ? coachingZones
    : [];

  return phaseConfig.items
    .map(([phaseKey, phaseLabel]) => {
      const imagePath = resolvePhaseImagePath(
        phaseImages,
        phaseKey
      );

      if (!imagePath) {
        return null;
      }

      const matchingZone =
        sourceZones.find((zone) => zone.imageKey === phaseKey) ||
        sourceZones.find(
          (zone) =>
            resolvePhaseImagePath(
              phaseImages,
              zone.imageKey
            ) === imagePath
        ) ||
        sourceZones.find(
          (zone) =>
            String(zone.title || "").toLowerCase() ===
            String(phaseLabel || "").toLowerCase()
        );

      return {
        id: `phase-${phaseKey}`,
        title: phaseLabel,
        imageKey: phaseKey,
        status: matchingZone?.status || "good",
        note:
          matchingZone?.note ||
          `Review the ${String(phaseLabel || "").toLowerCase()} position.`,
      };
    })
    .filter(Boolean);
};

const makeZone = (id, title, imageKey, status, note) => ({
  id,
  title,
  imageKey,
  status: status || "good",
  note: note || "",
});

const COACHING_TITLE_TO_BREAKDOWN = {
  Depth: ["depth", "bottom", "squat_depth"],
  Torso: ["torso", "torso_stack"],
  Knees: ["knees", "valgus"],
  Heels: ["heels"],
  Neck: ["neck", "head_position"],
  "Front Rack": ["front_rack"],
  "Bar Position": ["bar_position"],
  "Overhead Stability": ["overhead"],
  "Bar Stack": ["bar_path"],
  "First Pull": ["first_pull"],
  Extension: ["extension"],
  Turnover: ["turnover"],
  Catch: ["catch", "overhead_catch", "split_catch"],
  "Overhead Catch": ["overhead_catch", "catch"],
  Stability: ["stability"],
  "Bar Path": ["bar_path"],
  Dip: ["dip"],
  Drive: ["drive"],
  "Split Catch": ["split_catch"],
  Lockout: ["lockout", "active_finish"],
  "Leg Drive": ["leg_drive", "legs"],
  "Back Position": ["back"],
  "Hip Hinge": ["hinge"],
  Control: ["control"],
  Pull: ["pull", "range"],
  Transition: ["transition"],
  Support: ["support"],
  "Body Line": ["body_line"],
  Arch: ["arch"],
  Elbows: ["elbows"],
  Timing: ["timing"],
};

const zonesFromCoaching = (
  coaching,
  imageKeyByTitle = {},
  defaultImageKey = "setup",
  breakdown = {}
) => {
  if (!coaching?.sections?.length) return null;

  return coaching.sections.map((section) => {
    const breakdownKeys = COACHING_TITLE_TO_BREAKDOWN[section.title] || [];
    const breakdownStatus =
      breakdownKeys.length > 0
        ? pickBreakdown(breakdown, ...breakdownKeys)
        : "good";
    const coachingStatus = coachingSectionStatus(section) || "good";
    const status =
      breakdownStatus !== "good" ? breakdownStatus : coachingStatus;

    return {
      id: String(section.title || "zone")
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "_"),
      title: section.title,
      imageKey: imageKeyByTitle[section.title] || defaultImageKey,
      status,
      note: section.message || "",
    };
  });
};

const getPhaseConfig = (exerciseLabel) => {
  const label = String(exerciseLabel || "").toLowerCase().trim();

  if (label.includes("thruster")) {
    return {
      title: "Thruster Phase Review",
      text: "Start → Squat → Lockout",
      items: [
        ["setup", "Start"],
        ["dip", "Squat"],
        ["lockout", "Lockout"],
      ],
    };
  }

  if (
    label.includes("clean_and_jerk") ||
    label.includes("clean & jerk") ||
    label.includes("clean and jerk")
  ) {
    return {
      title: "Clean & Jerk Phase Review",
      text:
        "Setup → Clean Catch → Clean Recovery → Jerk Dip → Jerk Catch → Finish",
      items: [
        ["setup", "Setup"],
        ["clean_catch", "Clean Catch"],
        ["clean_recovery", "Clean Recovery"],
        ["jerk_dip", "Jerk Dip"],
        ["jerk_catch", "Jerk Catch"],
        ["finish", "Finish"],
      ],
    };
  }

  if (
    label.includes("split_jerk") ||
    label.includes("split jerk")
  ) {
    return {
      title: "Split Jerk Phase Review",
      text: "Setup → Dip → Drive → Catch → Finish",
      items: [
        ["setup", "Setup"],
        ["dip", "Dip"],
        ["drive", "Drive"],
        ["catch", "Catch"],
        ["finish", "Finish"],
      ],
    };
  }

  if (label === "clean" || label === "clean_only") {
    return {
      title: "Clean Phase Review",
      text:
        "Setup → First Pull → Transition → Power Position → Extension → Pull Under → Catch → Recovery",
      items: [
        ["setup", "Setup"],
        ["first_pull", "First Pull"],
        ["transition", "Transition"],
        ["power_position", "Power Position"],
        ["extension_v2", "Extension"],
        ["pull_under", "Pull Under"],
        ["catch", "Catch"],
        ["recovery", "Recovery"],
      ],
    };
  }

  if (
    !label.includes("push_press") &&
    !label.includes("push press") &&
    (
      label.includes("olympic") ||
      label.includes("snatch") ||
      label.includes("jerk")
    )
  ) {
    return {
      title: "Olympic Lift Phase Review",
      text: "Setup → First Pull → Pull Under → Catch → Lockout",
      items: [
        ["setup", "Setup"],
        ["first_pull", "First Pull"],
        ["extension", "Pull Under"],
        ["catch", "Catch"],
        ["finish", "Lockout"],
      ],
    };
  }

  if (
    label.includes("push_press") ||
    label.includes("push press")
  ) {
    return {
      title: "Push Press Phase Review",
      text: "Setup → Dip → Drive → Lockout",
      items: [
        ["setup", "Setup"],
        ["dip", "Dip"],
        ["drive", "Drive"],
        ["lockout", "Lockout"],
      ],
    };
  }

  if (
    label.includes("strict_press") ||
    label.includes("strict press") ||
    (label.includes("strict") && label.includes("press"))
  ) {
    return {
      title: "Strict Press Phase Review",
      text: "Setup → Press → Lockout",
      items: [
        ["setup", "Setup"],
        ["press", "Press"],
        ["lockout", "Lockout"],
      ],
    };
  }

  if (label.includes("squat")) {
    return {
      title: "Squat Phase Review",
      text: "Setup → Descent → Bottom → Ascent → Lockout",
      items: [
        ["setup", "Setup"],
        ["descent", "Descent"],
        ["bottom", "Bottom"],
        ["ascent", "Ascent"],
        ["lockout", "Lockout"],
      ],
    };
  }

  if (label.includes("bench")) {
    return {
      title: "Bench Press Phase Review",
      text: "Setup → Descent → Bottom → Press → Lockout",
      items: [
        ["setup", "Setup"],
        ["descent", "Descent"],
        ["bottom", "Bottom"],
        ["press", "Press"],
        ["lockout", "Lockout"],
      ],
    };
  }

  if (label.includes("deadlift")) {
    return {
      title: "Deadlift Phase Review",
      text: "Setup → Pull → Mid → Lockout",
      items: [
        ["setup", "Setup"],
        ["pull", "Pull"],
        ["mid", "Mid Pull"],
        ["lockout", "Lockout"],
      ],
    };
  }

  if (label.includes("muscle")) {
    return {
      title: label.includes("ring")
        ? "Ring Muscle-Up Phase Review"
        : "Bar Muscle-Up Phase Review",
      text: "Hang → Pull → Transition → Dip → Lockout → Finish",
      items: [
        ["hang", "Hang"],
        ["pull", "Pull"],
        ["transition", "Transition"],
        ["dip", "Dip"],
        ["lockout", "Lockout"],
        ["finish", "Finish"],
      ],
    };
  }

  if (
    label.includes("handstand push") ||
    label.includes("handstand_push")
  ) {
    return {
      title: "Handstand Push-Up Phase Review",
      text: "Setup → Descent → Bottom → Ascent",
      items: [
        ["setup", "Setup"],
        ["descent", "Descent"],
        ["bottom", "Bottom"],
        ["ascent", "Ascent"],
      ],
    };
  }

  if (
    label.includes("push up") ||
    label.includes("push-up") ||
    label.includes("push_up")
  ) {
    return {
      title: "Push-Up Phase Review",
      text: "Setup → Descent → Bottom → Ascent → Lockout",
      items: [
        ["setup", "Setup"],
        ["descent", "Descent"],
        ["bottom", "Bottom"],
        ["ascent", "Ascent"],
        ["lockout", "Lockout"],
      ],
    };
  }

  if (
    label.includes("pull up") ||
    label.includes("pull-up") ||
    label.includes("pull_up")
  ) {
    return {
      title: "Pull-Up Phase Review",
      text: "Hang → Pull → Top → Descent",
      items: [
        ["hang", "Hang"],
        ["pull", "Pull"],
        ["top", "Top"],
        ["descent", "Descent"],
      ],
    };
  }

  if (label.includes("burpee")) {
    return {
      title: "Burpee Phase Review",
        text: "Hands Down → Bottom → Jump In → Jump",
      items: [
          ["hands_down", "Hands Down"],
          ["bottom", "Bottom"],
          ["jump_in", "Jump In"],
          ["jump", "Jump"],
      ],
    };
  }

  return {
    title: "Movement Phase Review",
    text: "Setup → Movement → Finish",
    items: [
      ["setup", "Setup"],
      ["finish", "Finish"],
    ],
  };
};

const getInteractiveZones = (result) => {
  if (!result?.rep_feedback?.length) return [];

  const label = String(result?.exercise_label || "").toLowerCase();
  const bestRep = getBestRep(result.rep_feedback);
  if (!bestRep) return [];

  const breakdown = bestRep.breakdown || {};
  const coaching = bestRep.coaching;

  const status = (breakdownKeys, coachingTitles = []) =>
    resolveZoneStatus({ breakdown, coaching, breakdownKeys, coachingTitles });

  const note = (fallback, coachingTitles = []) =>
    resolveZoneNote({ coaching, coachingTitles, fallback, bestRep });

  if (label.includes("thruster")) {
    return [
      makeZone("squat_depth", "Squat Depth", "dip", status(["squat_depth"]), note("Use a full squat before driving the bar overhead.")),
      makeZone("torso", "Torso", "dip", status(["torso_stack"]), note("Stay tall through the squat and drive straight overhead.")),
      makeZone("lockout", "Lockout", "lockout", status(["lockout", "active_finish"]), note("Finish fully locked out overhead.")),
      makeZone("bar_path", "Bar Path", "drive", status(["bar_path"]), note("Keep the bar close and drive straight overhead.")),
    ];
  }

  if (label.includes("burpee")) {
    return [
        makeZone("hands_down", "Hands Down", "hands_down", status(["hands_down"]), note("Place both hands firmly on the floor.")),
        makeZone("bottom", "Bottom", "bottom", status(["plank", "bottom"]), note("Keep your body controlled through the floor position.")),
        makeZone("jump_in", "Jump In", "jump_in", status(["jump_in"]), note("Bring your feet underneath you efficiently.")),
        makeZone("jump", "Jump", "jump", status(["stand", "finish"]), note("Finish with a strong vertical jump.")),
    ];
  }

  if (label.includes("muscle")) {
    return [
      makeZone("pull", "Pull", "pull", status(["pull"]), note("Pull high before starting the transition.")),
      makeZone("transition", "Transition", "transition", status(["transition"]), note("Turn over aggressively and keep the rings or bar close.")),
      makeZone("support", "Support", "dip", status(["support"]), note("Stabilize above the bar or rings before finishing.")),
      makeZone("lockout", "Lockout", "lockout", status(["lockout"]), note("Finish tall with strong locked-out arms.")),
    ];
  }

  if (label.includes("handstand push")) {
    return [
      makeZone("depth", "Depth", "bottom", status(["depth", "bottom"]), note("Lower your head toward the floor under control.")),
      makeZone("body_line", "Body Line", "descent", status(["body_line"]), note("Keep your body stacked and avoid arching or sagging.")),
      makeZone("control", "Control", "ascent", status(["control", "range"]), note("Press smoothly away from the floor.")),
    ];
  }

  if (label.includes("push_up") || label.includes("push-up") || label.includes("push up")) {
    return [
      makeZone("depth", "Depth", "bottom", status(["bottom", "depth", "range"]), note("Lower your chest closer to the floor.")),
      makeZone("body_line", "Body Line", "descent", status(["body_line"]), note("Keep shoulders, hips, and ankles aligned.")),
      makeZone("lockout", "Lockout", "lockout", status(["lockout"]), note("Finish with elbows nearly straight.")),
      makeZone("range", "Range", "ascent", status(["range"]), note("Move through a full controlled range of motion.")),
    ];
  }

  if (label.includes("clean_and_jerk") || label.includes("clean & jerk") || label.includes("clean and jerk")) {
    const cleanBreakdown = breakdown.clean || breakdown;
    const jerkBreakdown = breakdown.jerk || {};
    return [
      makeZone("clean_catch", "Clean Catch", "clean_catch", status(["catch"], ["Catch"]), note("Receive the clean in a strong front rack.", ["Catch"])),
      makeZone("front_rack", "Front Rack", "clean_recovery", status(["front_rack"], ["Front Rack"]), note("Stand up smoothly from the clean before initiating the jerk.", ["Front Rack"])),
      makeZone("jerk_dip", "Jerk Dip", "jerk_dip", pickBreakdown(jerkBreakdown, "dip") !== "good" ? pickBreakdown(jerkBreakdown, "dip") : status(["dip"], ["Dip"]), note("Dip straight down with a vertical torso.", ["Dip"])),
      makeZone("jerk_lockout", "Jerk Lockout", "jerk_catch", pickBreakdown(jerkBreakdown, "lockout") !== "good" ? pickBreakdown(jerkBreakdown, "lockout") : status(["lockout"], ["Lockout"]), note("Catch locked out overhead with control.", ["Lockout"])),
      makeZone("bar_path", "Bar Path", "finish", pickBreakdown(cleanBreakdown, "bar_path"), note("Keep the bar close through the clean and jerk.")),
    ];
  }

  if (label.includes("split_jerk") || label.includes("split jerk")) {
    const fromCoaching = zonesFromCoaching(coaching, {
      Setup: "setup", Dip: "dip", Drive: "drive", "Split Catch": "catch",
      Lockout: "finish", Torso: "catch", "Bar Path": "drive",
    }, "setup", breakdown);
    if (fromCoaching?.length) return fromCoaching;
    return [
      makeZone("dip", "Dip", "dip", status(["dip"]), note("Dip straight down with a vertical torso.")),
      makeZone("drive", "Drive", "drive", status(["drive"]), note("Drive aggressively through the legs.")),
      makeZone("split_catch", "Split Catch", "catch", status(["split_catch"]), note("Catch locked out overhead in a strong split.")),
      makeZone("lockout", "Lockout", "finish", status(["lockout"]), note("Recover under control and stabilize overhead.")),
      makeZone("torso_stack", "Torso Stack", "catch", status(["torso_stack"]), note("Keep ribs stacked and torso vertical under the bar.")),
      makeZone("bar_path", "Bar Path", "drive", status(["bar_path"]), note("Drive the bar straight up overhead.")),
    ];
  }

  if (label.includes("snatch")) {
    const fromCoaching = zonesFromCoaching(coaching, {
      "First Pull": "first_pull", Extension: "extension", Turnover: "catch",
      "Overhead Catch": "catch", Catch: "catch", Stability: "finish", "Bar Path": "first_pull",
    }, "setup", breakdown);
    if (fromCoaching?.length) return fromCoaching;
    return [
      makeZone("first_pull", "First Pull", "first_pull", status(["first_pull"]), note("Keep the bar close as it passes the knees.")),
      makeZone("extension", "Extension", "extension", status(["extension"]), note("Drive tall through the legs and hips.")),
      makeZone("turnover", "Turnover", "catch", status(["turnover"]), note("Pull yourself under the bar aggressively.")),
      makeZone("overhead_catch", "Overhead Catch", "catch", status(["overhead_catch", "catch"]), note("Receive the bar in a strong overhead position.")),
      makeZone("stability", "Stability", "finish", status(["stability"]), note("Stabilize the bar overhead before standing.")),
      makeZone("bar_path", "Bar Path", "first_pull", status(["bar_path"]), note("Keep the bar closer to your body during the pull.")),
    ];
  }

  if (label.includes("clean") && !label.includes("jerk")) {
    const fromCoaching = zonesFromCoaching(coaching, {
      "First Pull": "first_pull", Extension: "extension", Turnover: "pull_under",
      Catch: "catch", "Front Rack": "catch", "Bar Path": "first_pull",
    }, "setup", breakdown);
    if (fromCoaching?.length) return fromCoaching;
    return [
      makeZone("first_pull", "First Pull", "first_pull", status(["first_pull"]), note("Keep the bar close as it passes the knees.")),
      makeZone("extension", "Extension", "extension", status(["extension"]), note("Drive tall through the legs and hips.")),
      makeZone("turnover", "Turnover", "pull_under", status(["turnover"]), note("Rotate the elbows faster into the rack.")),
      makeZone("catch", "Catch", "catch", status(["catch"]), note("Receive the bar under control.")),
      makeZone("front_rack", "Front Rack", "catch", status(["front_rack"]), note("Drive elbows high and keep the bar on your shoulders.")),
      makeZone("bar_path", "Bar Path", "first_pull", status(["bar_path"]), note("Keep the bar close through the pull and turnover.")),
    ];
  }

  if (label.includes("push_press") || label.includes("push press")) {
    return [
      makeZone("dip", "Dip", "dip", status(["dip"]), note("Use a vertical dip and drive straight through the bar.")),
      makeZone("timing", "Timing", "drive", status(["timing"]), note("Connect the leg drive to the press without pressing too early.")),
      makeZone("lockout", "Lockout", "lockout", status(["lockout", "active_finish"]), note("Finish with elbows fully extended overhead.")),
      makeZone("bar_path", "Bar Path", "drive", status(["bar_path"]), note("Keep the bar close and drive straight overhead.")),
      makeZone("valgus", "Knees", "dip", status(["valgus"]), note("Drive knees out and keep a stable dip.")),
    ];
  }

  if (label.includes("strict_press") || label.includes("strict press") || (label.includes("strict") && label.includes("press"))) {
    return [
      makeZone("leg_drive", "Leg Drive", "setup", status(["leg_drive"]), note("Keep your knees locked and press without dipping.")),
      makeZone("torso_stack", "Torso Stack", "press", status(["torso_stack"]), note("Brace ribs down and avoid overextending your lower back.")),
      makeZone("lockout", "Lockout", "lockout", status(["lockout", "active_finish"]), note("Finish stacked overhead with strong elbow extension.")),
      makeZone("bar_path", "Bar Path", "press", status(["bar_path"]), note("Press straight up and move your head through as the bar passes.")),
    ];
  }

  if (label.includes("squat")) {
    const fromCoaching = zonesFromCoaching(coaching, {
      Depth: "bottom", Torso: "descent", Knees: "bottom", Heels: "lockout", Neck: "setup",
      "Front Rack": "bottom", "Bar Position": "descent", "Overhead Stability": "lockout", "Bar Stack": "descent",
    }, "bottom", breakdown);
    if (fromCoaching?.length) return fromCoaching;

    const zones = [
      makeZone("depth", "Depth", "bottom", status(["depth"]), note("Reach clear depth while keeping control.")),
      makeZone("torso", "Torso", "descent", status(["torso"]), note("Keep your chest tall and avoid folding forward.")),
      makeZone("knees", "Knees", "bottom", status(["knees"]), note("Drive knees out and keep them tracking over toes.")),
      makeZone("heels", "Heels", "lockout", status(["heels"]), note("Keep your heels planted and drive through midfoot.")),
      makeZone("neck", "Neck", "setup", status(["neck"]), note("Keep your head aligned with your torso.")),
    ];
    if (breakdown.front_rack) zones.push(makeZone("front_rack", "Front Rack", "bottom", status(["front_rack"]), note("Drive elbows higher to keep the bar secure.")));
    if (breakdown.bar_position) zones.push(makeZone("bar_position", "Bar Position", "descent", status(["bar_position"]), note("Keep the bar close to your throat and elbows pointed forward.")));
    if (breakdown.overhead) zones.push(makeZone("overhead", "Overhead Stability", "lockout", status(["overhead"]), note("Lock the bar directly over midfoot and stay stacked.")));
    if (breakdown.bar_path) zones.push(makeZone("bar_path", "Bar Stack", "descent", status(["bar_path"]), note("Prevent forward drift — keep the bar over midfoot.")));
    return zones;
  }

  if (label.includes("bench")) {
    return [
      makeZone("depth", "Depth", "bottom", status(["depth"]), note("Lower the bar under control toward your chest.")),
      makeZone("elbows", "Elbows", "bottom", status(["elbows"]), note("Keep elbows controlled without excessive flare.")),
      makeZone("lockout", "Lockout", "lockout", status(["lockout"]), note("Finish with arms fully extended.")),
      makeZone("arch", "Arch", "press", status(["arch"]), note("Keep a controlled arch without losing ribcage position.")),
      makeZone("legs", "Leg Drive", "press", status(["legs"]), note("Keep feet planted and drive through your legs.")),
    ];
  }

  if (label.includes("deadlift")) {
    return [
      makeZone("back", "Back Position", "pull", status(["back"]), note("Brace hard and keep a neutral spine.")),
      makeZone("hinge", "Hip Hinge", "mid", status(["hinge"]), note("Push hips back and keep tension through the pull.")),
      makeZone("bar_path", "Bar Path", "mid", status(["bar_path"]), note("Keep the bar close to your body.")),
      makeZone("lockout", "Lockout", "lockout", status(["lockout"]), note("Finish tall with hips and knees extended.")),
      makeZone("knees", "Knees", "mid", status(["knees"]), note("Keep knees tracking and avoid excessive bend.")),
      makeZone("control", "Control", "lockout", status(["control"]), note("Lower the bar with control on the way down.")),
    ];
  }

  if (label.includes("pull") && !label.includes("push")) {
    return [
      makeZone("hang", "Hang", "hang", status(["range"]), note("Start from a controlled dead hang.")),
      makeZone("pull", "Pull", "pull", status(["range"]), note("Pull strongly with control.")),
      makeZone("top", "Top", "top", status(["top"]), note("Finish high with chin near or above the bar.")),
      makeZone("descent", "Descent", "descent", status(["control"]), note("Lower under control without beginning another rep.")),
    ];
  }

  return [
    makeZone("back", "Back Position", "pull", status(["back"]), note("Brace hard and keep a neutral spine.")),
    makeZone("hinge", "Hip Hinge", "mid", status(["hinge"]), note("Push hips back and keep tension through the pull.")),
    makeZone("bar_path", "Bar Path", "mid", status(["bar_path"]), note("Keep the bar close to your body.")),
    makeZone("lockout", "Lockout", "lockout", status(["lockout"]), note("Finish tall with hips and knees extended.")),
    makeZone("knees", "Knees", "mid", status(["knees"]), note("Keep knees tracking and avoid excessive bend.")),
    makeZone("control", "Control", "lockout", status(["control"]), note("Lower the bar with control on the way down.")),
  ];
};

const QUEUE_KEY = "formcheck_pending_videos";

const HISTORY_KEY = "formcheck_analysis_history";

const savePendingVideo = async (video) => {
  const id = Date.now().toString();

  const ext = video.uri?.split(".").pop() || "mov";
  const localUri = `${FileSystem.documentDirectory}pending_${id}.${ext}`;

  await FileSystem.copyAsync({
    from: video.uri,
    to: localUri,
  });

  const item = {
    id,
    uri: localUri,
    name: video.name || video.fileName || `Workout ${id}`,
    createdAt: new Date().toISOString(),
    status: "pending",
  };

  const existing =
    JSON.parse(await AsyncStorage.getItem(QUEUE_KEY)) || [];

  const updated = [item, ...existing];

  await AsyncStorage.setItem(
    QUEUE_KEY,
    JSON.stringify(updated)
  );

  return item;
};

const loadPendingVideos = async (setter) => {
  const existing =
    JSON.parse(await AsyncStorage.getItem(QUEUE_KEY)) || [];

  setter(existing);
};

const saveAnalysisHistory = async (analysis) => {
  try {
    const item = {
      id: Date.now().toString(),
      createdAt: new Date().toISOString(),
      exercise_label: analysis.exercise_label,
      confidence: analysis.confidence,
      set_summary: analysis.set_summary || {},
      rep_feedback: analysis.rep_feedback || [],
      coaching_zones: analysis.coaching_zones || [],
    };

    const existing =
      JSON.parse(await AsyncStorage.getItem(HISTORY_KEY)) || [];

    const updated = [item, ...existing].slice(0, 100);

    await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(updated));
  } catch (err) {
    console.log("SAVE HISTORY ERROR:", err);
  }
};

export default function App() {
  const [video, setVideo] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [visualsLoading, setVisualsLoading] = useState(false);
  const [selectedZone, setSelectedZone] = useState(null);
  const [pendingVideos, setPendingVideos] = useState([]);
  const [overlayUrl, setOverlayUrl] = useState(null);
  const [overlayLoading, setOverlayLoading] = useState(false);
  const [overlayProgress, setOverlayProgress] = useState("");
  const [overlayStatus, setOverlayStatus] = useState("idle");

  useEffect(() => {
    loadPendingVideos(setPendingVideos);
  }, []);

  const reset = () => {
    setResult(null);
    setSelectedZone(null);
    setVisualsLoading(false);
  };

  const recordWithCamera = async () => {
    reset();

    const permission = await ImagePicker.requestCameraPermissionsAsync();

    if (!permission.granted) {
      Alert.alert("Camera permission required");
      return;
    }

    const res = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      quality: 1,
      videoMaxDuration: 60,
    });

    if (!res.canceled) {
      const a = res.assets[0];

      const selected = {
        uri: a.uri,
        file: a.file,
        name: a.fileName || a.name || "library.mov",
        type: a.mimeType || a.type || "video/quicktime",
      };

      setVideo(selected);

      await savePendingVideo(selected);
      await loadPendingVideos(setPendingVideos);
    }
  };

  const pickWebVideoFile = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "video/*";

    input.onchange = (e) => {
      const file = e.target.files?.[0];

      if (!file) return;

      setResult(null);
      setSelectedZone(null);
      setOverlayUrl(null);

      setVideo({
        file,
        uri: URL.createObjectURL(file),
        name: file.name || "upload.mov",
        type: file.type || "video/mp4",
      });
    };

    input.click();
  };

  const pickFromLibrary = async () => {
    reset();

    if (Platform.OS === "web") {
      pickWebVideoFile();
      return;
    }

    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (!permission.granted) {
      Alert.alert("Library permission required");
      return;
    }

    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      quality: 1,
    });

    if (!res.canceled) {
      const a = res.assets[0];

      const selected = {
        uri: a.uri,
        name: a.fileName || a.name || "library.mov",
        type: a.mimeType || a.type || "video/quicktime",
      };

      setVideo(selected);

      await savePendingVideo(selected);
      await loadPendingVideos(setPendingVideos);
    }
  };

  const pickFromCloud = async () => {
    reset();

    if (Platform.OS === "web") {
      pickWebVideoFile();
      return;
    }

    const res = await DocumentPicker.getDocumentAsync({
      type: "video/*",
      copyToCacheDirectory: true,
    });

    if (!res.canceled) {
      const a = res.assets[0];

      const selected = {
        uri: a.uri,
        name: a.name || "cloud.mov",
        type: a.mimeType || "video/quicktime",
      };

      setVideo(selected);

      await savePendingVideo(selected);
      await loadPendingVideos(setPendingVideos);
    }
  };

  const buildFormData = async (extra = {}) => {
    const formData = new FormData();

    if (Platform.OS === "web") {
      if (!video?.file) {
        throw new Error(
          "No browser file found. Please choose the video again.",
        );
      }

      formData.append("file", video.file, video.name || "upload.mov");
    } else {
      formData.append("file", {
        uri: video.uri,
        name: video.name || "upload.mov",
        type: video.type || "video/mp4",
      });
    }

    Object.entries(extra).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        formData.append(key, value);
      }
    });

    return formData;
  };

  const generateVisuals = async (analysisResult) => {
    if (visualsLoading) return;
    const hasUrls =
      analysisResult?.phase_images &&
      Object.values(analysisResult.phase_images).some(isPhaseImageUrl);
    if (hasUrls) return;

    try {
      console.log("GENERATE VISUALS STARTED");

      setVisualsLoading(true);

      const bestRep = getBestRep(analysisResult?.rep_feedback || []);

      const visualsRes = await fetch(`${API_URL}/generate_visuals`, {
        method: "POST",
        body: await buildFormData({
          rep_json: bestRep ? JSON.stringify(bestRep) : null,
          exercise_label: analysisResult?.exercise_label || "",
        }),
      });

      const visualsText = await visualsRes.text();

      let visualsData = {};
      try {
        visualsData = JSON.parse(visualsText);
      } catch {
        visualsData = {
          message:
            visualsText || "Visual generation returned non-JSON response",
        };
      }

      console.log("VISUALS RESPONSE:", visualsData);
      console.log("VISUALS STATUS:", visualsRes.status);

      if (!visualsRes.ok) {
        throw new Error(
          visualsData.detail ||
            visualsData.message ||
            "Visual generation request failed",
        );
      }

      if (visualsData.overlay_video_url) {
        setOverlayUrl(fullUrl(visualsData.overlay_video_url));
      }

      setResult((prev) => ({
        ...prev,
        overlay_video_url:
          visualsData.overlay_video_url || prev?.overlay_video_url || null,
        phase_images: visualsData.phase_images || {},
        visuals_error: visualsData.visuals_error || null,
      }));
    } catch (err) {
      console.log("VISUALS ERROR:", err);

      setResult((prev) => ({
        ...prev,
        visuals_error: err.message,
      }));
    } finally {
      setVisualsLoading(false);
    }
  };

  const generateOverlay = async () => {
    console.log("GENERATE OVERLAY CLICKED", {
      hasVideo: !!video,
      hasResult: !!result,
      exercise: result?.exercise_label,
    });

    try {
      setOverlayLoading(true);
      setOverlayProgress("Generating overlay...");

      const bestRep = getBestRep(result?.rep_feedback || []);

      const overlayRes = await fetch(`${API_URL}/generate_overlay`, {
        method: "POST",
        body: await buildFormData({
          rep_json: bestRep ? JSON.stringify(bestRep) : null,
          exercise_label: result?.exercise_label || "",
        }),
      });

      const overlayText = await overlayRes.text();

      let overlayData = {};
      try {
        overlayData = JSON.parse(overlayText);
      } catch {
        overlayData = { message: overlayText || "Overlay returned non-JSON response" };
      }

      console.log("OVERLAY STATUS:", overlayRes.status);
      console.log("OVERLAY RESPONSE:", overlayData);

      if (!overlayRes.ok || !overlayData.overlay_video_url) {
        throw new Error(
          overlayData.overlay_error ||
          overlayData.detail ||
          overlayData.message ||
          "Overlay generation failed"
        );
      }

      setOverlayProgress("Overlay ready!");
      setOverlayUrl(fullUrl(overlayData.overlay_video_url));

      setResult((prev) => ({
        ...prev,
        overlay_video_url: overlayData.overlay_video_url,
        overlay_error: null,
      }));
    } catch (err) {
      console.log("OVERLAY ERROR:", err);

      setResult((prev) => ({
        ...prev,
        overlay_error: err.message,
      }));
    } finally {
      setOverlayLoading(false);
      setOverlayProgress("");
    }
  };

  const pollOverlay = async (jobId) => {
    const interval = setInterval(async () => {
      const res = await fetch(`${API_URL}/overlay_status/${jobId}`);
      const data = await res.json();

      if (data.status === "done") {
        clearInterval(interval);
        setOverlayUrl(data.url);
        setOverlayStatus("done");
      }

      if (data.status === "failed") {
        clearInterval(interval);
        setOverlayStatus("failed");
      }
    }, 2000);
  };

  const analyzeVideo = async () => {
  if (!video) {
    Alert.alert("Pick a video first");
    return;
  }

  setLoading(true);
  setResult(null);
  setSelectedZone(null);
  setVisualsLoading(false);

  try {
    const res = await fetch(`${API_URL}/analyze`, {
      method: "POST",
      body: await buildFormData(),
    });

    const data = await res.json();

    console.log("ANALYZE RESPONSE:", data);

    if (!res.ok) {
      throw new Error(
        data?.detail ||
        data?.message ||
        "Analyze request failed"
      );
    }
      setSelectedZone(null);
      setOverlayUrl(null);

      const analysisResult = {
        ...data,
        phase_images: data.phase_images || {},
      };

      setResult(analysisResult);

      if (analysisResult.overlay_video_url) {
        setOverlayUrl(fullUrl(analysisResult.overlay_video_url));
      }

      await saveAnalysisHistory(analysisResult);

  } catch (err) {
    console.error(err);
    setResult({ error: true, message: err.message });
  } finally {
    setLoading(false);
  }
};

  const reps = result?.rep_feedback || [];

  const avgScore = useMemo(() => {
    if (!reps.length) return null;

    const avg =
      reps.reduce((sum, rep) => sum + Number(rep.score || 0), 0) / reps.length;

    return Number(avg.toFixed(1));
  }, [reps]);

  const displayScore = avgScore !== null ? Math.round(avgScore * 10) : null;
  const bestRep = getBestRep(reps);

  // biggest_fix is not returned by the backend — derive it from rep feedback
  const biggestFix =
    result?.set_summary?.biggest_fix ||
    bestRep?.coaching?.priority ||
    bestRep?.issues?.[0] ||
    bestRep?.feedback?.[0] ||
    "Upload a clear side-angle video for analysis.";

  const phaseConfig = getPhaseConfig(result?.exercise_label);
  const rawPhaseImages = result?.phase_images || {};
  const phaseImages = Object.fromEntries(
    Object.entries(rawPhaseImages).filter(([_, value]) =>
      isPhaseImageUrl(value)
    )
  );
  const hasPhaseImageUrls = Object.keys(phaseImages).length > 0;
  const rawZones = getInteractiveZones(result);

  // Interactive Coaching Map mirrors Phase Review exactly:
  // same order, same labels, and same image keys/files.
  const zones = buildPhaseMatchedZones(
    rawZones,
    phaseConfig,
    phaseImages
  );

  const activeZone = selectedZone
    ? zones.find((zone) => zone.id === selectedZone.id) || zones[0]
    : zones[0];

  const activeImagePath =
    resolvePhaseImagePath(phaseImages, activeZone?.imageKey);

  const activeImageUrl = fullUrl(activeImagePath);
  const overlayPlayer = useVideoPlayer(overlayUrl, (player) => {
    player.loop = false;
  });

  const buildCoachSummary = (result) => {
    if (!result) return "";

    const bestRep = getBestRep(result.rep_feedback || []);
    if (!bestRep) return "";

    const issues = bestRep.issues || [];
    const breakdown = bestRep.breakdown || {};
    const coaching = bestRep.coaching;

    // If the backend provided coaching data (Olympic lifts), use it directly.
    // This gives real per-phase messages instead of reconstructed cues.
    if (coaching?.priority && coaching?.sections?.length > 0) {
      let text = coaching.priority;

      // Add messages for any non-good sections
      const warnings = coaching.sections.filter(
        (s) => s.status === "warning" || s.status === "poor"
      );

      if (warnings.length > 0) {
        text += " Focus on: " + warnings.map((s) => s.message).join(" ");
      }

      // Append any issues the backend flagged
      if (issues.length > 0) {
        const issueText = issues.slice(0, 2).join(" ");
        if (!text.includes(issueText)) {
          text += " " + issueText;
        }
      }

      return text;
    }

    // Fallback: reconstruct from breakdown (squats, deadlifts, bench press)
    let intro = "Solid rep overall.";
    let body = [];
    let cues = [];

    if (bestRep.score >= 9) {
      intro = "Great rep — very strong execution.";
    } else if (bestRep.score >= 7) {
      intro = "Good rep overall, but there are a couple things to clean up.";
    } else {
      intro = "This rep needs some work.";
    }

    if (issues.length > 0) body.push(issues[0]);

    if (breakdown.knees === "poor")
      cues.push("Drive your knees out and keep them tracking over your toes.");
    if (breakdown.depth === "borderline")
      cues.push("Sit slightly deeper while keeping your chest up.");
    if (breakdown.torso === "poor")
      cues.push("Keep your chest tall and avoid leaning forward.");
    if (breakdown.heels === "poor")
      cues.push("Keep your weight through your mid-foot and heels.");
    if (breakdown.first_pull === "poor")
      cues.push("Keep the bar close and maintain tension off the floor.");
    if (breakdown.extension === "poor")
      cues.push("Drive tall through your hips — full extension before the pull.");
    if (breakdown.catch === "poor")
      cues.push("Lock in under the bar — fast elbows into a stable receiving position.");
    if (breakdown.lockout === "incomplete" || breakdown.lockout === "poor" || breakdown.lockout === "soft" || breakdown.lockout === "short")
      cues.push("Punch aggressively into lockout and hold the overhead position.");
    if (breakdown.back === "poor" || breakdown.back === "fair")
      cues.push("Brace your core and keep a neutral spine throughout the pull.");
    if (breakdown.hinge === "poor")
      cues.push("Push your hips back and load the hamstrings before driving up.");
    if (breakdown.bar_path === "drifting" || breakdown.bar_path === "poor")
      cues.push("Keep the bar close to your body through the full range of motion.");
    if (breakdown.elbows === "severe_flare" || breakdown.elbows === "borderline")
      cues.push("Tuck elbows slightly and control the bar path.");
    if (breakdown.arch === "excessive")
      cues.push("Keep a controlled arch without losing ribcage position.");
    if (breakdown.legs === "weak" || breakdown.legs === "unknown")
      cues.push("Keep feet planted and drive through your legs.");
    if (breakdown.body_line === "sagging" || breakdown.body_line === "borderline")
      cues.push("Keep your body in a straight line from head to heels.");
    if (breakdown.pull === "short" || breakdown.transition === "slow")
      cues.push("Pull higher and turn over the bar more aggressively.");
    if (breakdown.plank === "sagging" || breakdown.plank === "borderline")
      cues.push("Brace your core and keep a tight plank position.");

    if (bestRep.feedback?.length > 1) {
      body.push(...bestRep.feedback.slice(1, 3));
    }

    let text = intro;
    if (body.length > 0) text += " " + body.join(" ");
    if (cues.length > 0) text += " Focus on this next: " + cues.join(" ");

    return text;
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.hero}>
          <View>
            <Text style={styles.eyebrow}>AI Movement Coach</Text>
            <Text style={styles.title}>FormCheck AI</Text>
            <Text style={styles.subtitle}>
              Upload a lift. Get scoring, coaching zones, phase images, and
              replay.
            </Text>
          </View>

          <View style={styles.logoBubble}>
            <Text style={styles.logoText}>AI</Text>
          </View>
        </View>

        <View style={styles.actionRow}>
          <TouchableOpacity
            style={styles.primaryButton}
            onPress={recordWithCamera}
          >
            <Text style={styles.primaryButtonText}>Record</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.secondaryButton}
            onPress={pickFromLibrary}
          >
            <Text style={styles.secondaryButtonText}>Library</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.secondaryButton}
            onPress={pickFromCloud}
          >
            <Text style={styles.secondaryButtonText}>Files</Text>
          </TouchableOpacity>
        </View>

          {video && (
            <View style={styles.selectedCard}>
              <Text style={styles.cardLabel}>Selected Video</Text>
              <Text style={styles.selectedName}>{video.name}</Text>
            </View>
          )}

        {pendingVideos.length > 0 && (
          <View style={styles.card}>
            <Text style={styles.sectionTitle}>Pending Videos</Text>
            <Text style={styles.sectionSub}>
              Saved locally for later analysis
            </Text>

            {pendingVideos.map((v) => (
              <TouchableOpacity
                key={v.id}
                style={styles.selectedCard}
                onPress={() => setVideo(v)}
              >
                <Text style={styles.cardLabel}>Pending</Text>
                <Text style={styles.selectedName}>{v.name}</Text>
              </TouchableOpacity>
            ))}
          </View>
        )}

        <TouchableOpacity
          style={[
            styles.analyzeButton,
            (loading || visualsLoading) && styles.disabledButton,
          ]}
          onPress={analyzeVideo}
          disabled={loading || visualsLoading}
        >
          {loading ? (
            <View style={styles.loadingBlock}>
              <ActivityIndicator color="#020617" />
              <Text style={styles.analyzeButtonText}>
                Analyzing movement...
              </Text>
            </View>
          ) : visualsLoading ? (
            <View style={styles.loadingBlock}>
              <ActivityIndicator color="#020617" />
              <Text style={styles.analyzeButtonText}>Building visuals...</Text>
            </View>
          ) : (
            <Text style={styles.analyzeButtonText}>Analyze Lift</Text>
          )}
        </TouchableOpacity>

        {result?.error && (
          <View style={styles.errorCard}>
            <Text style={styles.errorTitle}>Request Failed</Text>
            <Text style={styles.errorText}>{result.message}</Text>
          </View>
        )}

        {result && !result.error && reps.length === 0 && (
          <View style={styles.errorCard}>
            <Text style={styles.errorTitle}>Video Not Usable</Text>
            <Text style={styles.errorText}>
              Camera angle is too close or unclear for reliable scoring.
            </Text>
            <Text style={styles.errorText}>
              Move farther back, record from the side, and keep the full body
              visible.
            </Text>
          </View>
        )}

        {result && !result.error && reps.length > 0 && (
          <>
            <View style={styles.dashboardGrid}>
              <View style={styles.scoreCard}>
                <Text style={styles.cardLabel}>Overall Score</Text>

                <View
                  style={[
                    styles.scoreCircle,
                    { borderColor: getScoreColor(avgScore || 0) },
                  ]}
                >
                  <Text style={styles.scoreBig}>{displayScore}</Text>
                  <Text style={styles.scoreSmall}>/100</Text>
                </View>

                <Text style={styles.exerciseName}>{formatLabel(result.exercise_label)}</Text>
                <Text style={styles.confidenceText}>
                  Confidence {Math.round(Number(result.confidence || 0) * 100)}%
                </Text>
              </View>

              <View style={styles.insightCard}>
                <Text style={styles.cardLabel}>Biggest Fix</Text>
                <Text style={styles.bigFix}>{biggestFix}</Text>

                <View style={styles.miniStats}>
                  <View style={styles.statPill}>
                    <Text style={styles.statNumber}>
                      {result?.set_summary?.detected_reps || reps.length}
                    </Text>
                    <Text style={styles.statLabel}>Reps</Text>
                  </View>

                  <View style={styles.statPill}>
                    <Text style={styles.statNumber}>
                      {result?.set_summary?.best_rep || bestRep?.rep || "-"}
                    </Text>
                    <Text style={styles.statLabel}>Best</Text>
                  </View>

                  <View style={styles.statPill}>
                    <Text style={styles.statNumber}>
                      {result?.set_summary?.worst_rep || "-"}
                    </Text>
                    <Text style={styles.statLabel}>Needs Work</Text>
                  </View>
                </View>
              </View>
            </View>

            {visualsLoading && (
              <View style={styles.warningCard}>
                <ActivityIndicator color="#86efac" />
                <Text style={styles.warningTitle}>
                  Coaching visuals are loading
                </Text>
                <Text style={styles.warningText}>
                  Scores are ready. Phase images and replay will appear next.
                </Text>
              </View>
            )}

            {result?.visuals_error && (
              <View style={styles.errorCard}>
                <Text style={styles.errorTitle}>Visuals Could Not Load</Text>
                <Text style={styles.errorText}>{result.visuals_error}</Text>
              </View>
            )}

            {
              <TouchableOpacity
                style={[
                  styles.analyzeButton,
                  (visualsLoading || hasPhaseImageUrls) &&
                    styles.disabledButton,
                ]}
                onPress={() => generateVisuals(result)}
                disabled={visualsLoading || hasPhaseImageUrls}
              >
                <Text style={styles.analyzeButtonText}>
                  {visualsLoading
                    ? "Generating Phase Review..."
                    : hasPhaseImageUrls
                      ? "Phase Review Generated"
                      : "Generate Phase Review"}
                </Text>
              </TouchableOpacity>
            }

            <View style={styles.card}>
              <Text style={styles.sectionTitle}>Rep Breakdown</Text>
              <Text style={styles.sectionSub}>
                {result?.set_summary?.trend || "Score trend across the set"}
              </Text>
              {reps.map((rep) => {
                const score = Number(rep.score || 0);
                const barWidth = `${Math.min(100, Math.max(8, score * 10))}%`;
                const repNotes = [
                  ...(rep.feedback || []),
                  ...(rep.issues || []),
                ].filter(Boolean);

                return (
                  <View key={`rep-${rep.rep}`} style={styles.repRow}>
                    <View style={styles.repTop}>
                      <Text style={styles.repLabel}>
                        Rep {rep.rep}
                        {rep.grade ? ` · ${rep.grade}` : ""}
                      </Text>
                      <Text style={styles.repScore}>{score.toFixed(1)}/10</Text>
                    </View>

                    <View style={styles.repBarTrack}>
                      <View
                        style={[
                          styles.repBarFill,
                          {
                            width: barWidth,
                            backgroundColor: getScoreColor(score),
                          },
                        ]}
                      />
                    </View>

                    {repNotes.length > 0 ? (
                      repNotes.map((line, idx) => (
                        <Text key={`rep-${rep.rep}-note-${idx}`} style={styles.repFeedback}>
                          {line}
                        </Text>
                      ))
                    ) : (
                      <Text style={styles.repFeedback}>Good rep.</Text>
                    )}
                  </View>
                );
              })}
            </View>

            <View style={styles.card}>
              <Text style={styles.sectionTitle}>Interactive Coaching Map</Text>
              <Text style={styles.sectionSub}>
                Tap a zone to review form cues and status colors from the analysis.
              </Text>

              <View style={styles.coachImageWrap}>
                {activeImageUrl ? (
                  <Image
                    key={`${activeZone?.id}-${activeImagePath}`}
                    source={{ uri: activeImageUrl }}
                    style={styles.coachImage}
                    resizeMode="contain"
                  />
                ) : (
                  <View style={styles.emptyImage}>
                    <Text style={styles.emptyImageText}>
                      {hasPhaseImageUrls
                        ? "Select a zone to preview the matching phase frame."
                        : "Generate phase review images to preview each zone."}
                    </Text>
                  </View>
                )}
              </View>

              <View style={styles.zoneGrid}>
                {zones.map((zone) => {
                  const isActive = activeZone?.id === zone.id;
                  const color = getStatusColor(zone.status);

                  return (
                    <TouchableOpacity
                      key={zone.id}
                      style={[
                        styles.zonePill,
                        isActive && styles.zonePillActive,
                        isActive && { borderColor: color },
                      ]}
                      onPress={() => setSelectedZone(zone)}
                    >
                      <View
                        style={[styles.statusDot, { backgroundColor: color }]}
                      />
                      <Text style={styles.zoneText}>{zone.title}</Text>
                    </TouchableOpacity>
                  );
                })}
              </View>

              {activeZone && (
                <View style={styles.coachingNote}>
                  <Text style={styles.noteTitle}>{activeZone.title}</Text>
                  <Text style={styles.noteStatus}>
                    Status: {formatLabel(activeZone.status)}
                  </Text>
                  <Text style={styles.noteText}>{activeZone.note}</Text>
                </View>
              )}
            </View>

            {bestRep?.visibility_notes?.length > 0 && (
              <View style={styles.warningCard}>
                <Text style={styles.warningTitle}>Visibility Notes</Text>
                {bestRep.visibility_notes.map((note, idx) => (
                  <Text key={`visibility-${idx}`} style={styles.warningText}>
                    {note}
                  </Text>
                ))}
              </View>
            )}

              {hasPhaseImageUrls && (
                <>
            <View style={styles.card}>
              <Text style={styles.sectionTitle}>{phaseConfig.title}</Text>
              <Text style={styles.sectionSub}>{phaseConfig.text}</Text>

              <ScrollView
                horizontal
                showsHorizontalScrollIndicator={false}
                contentContainerStyle={styles.phaseScroller}
              >
                {phaseConfig.items.map(([key, label]) => {
                  const path = resolvePhaseImagePath(phaseImages, key);
                  const url = fullUrl(path);

                  return (
                    <View key={key} style={styles.phaseCard}>
                      <View style={styles.phaseImageWrap}>
                        {url ? (
                          <Image
                            source={{ uri: url }}
                            style={styles.phaseImage}
                              resizeMode="contain"
                          />
                        ) : (
                          <View style={styles.emptyPhase}>
                            <Text style={styles.emptyPhaseText}>No image</Text>
                          </View>
                        )}
                      </View>
                      <Text style={styles.phaseLabel}>{label}</Text>
                    </View>
                  );
                })}
              </ScrollView>
            </View>

                </>
              )}

            <TouchableOpacity
              style={[
                styles.overlayButton,
                overlayLoading && styles.disabledButton,
              ]}
              onPress={generateOverlay}
              disabled={overlayLoading}
            >
              {overlayLoading ? (
                <View style={styles.loadingBlock}>
                  <ActivityIndicator color="#020617" />
                  <Text style={styles.analyzeButtonText}>
                    {overlayProgress || "Generating overlay..."}
                  </Text>
                </View>
              ) : (
                <Text style={styles.analyzeButtonText}>Generate Overlay</Text>
              )}
            </TouchableOpacity>

            {overlayUrl && (
              <View style={styles.overlayCard}>
                <Text style={styles.sectionTitle}>Coached Replay</Text>
                <Text style={styles.sectionSub}>
                  Overlay video with rep feedback and movement markers.
                </Text>

                <VideoView
                  player={overlayPlayer}
                  style={styles.videoPlayer}
                  allowsFullscreen
                  allowsPictureInPicture
                  nativeControls
                  contentFit="contain"
                />
              </View>
            )}

            <View style={styles.card}>
              <Text style={styles.sectionTitle}>Coach Summary</Text>

              <Text style={styles.coachText}>{buildCoachSummary(result)}</Text>
            </View>
          </>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#020617",
  },
  content: {
    padding: 18,
    paddingBottom: 42,
  },
  hero: {
    backgroundColor: "#0f172a",
    borderRadius: 28,
    padding: 22,
    borderWidth: 1,
    borderColor: "#1e293b",
    flexDirection: "row",
    justifyContent: "space-between",
    gap: 16,
    marginBottom: 16,
  },
  eyebrow: {
    color: "#86efac",
    fontSize: 13,
    fontWeight: "800",
    letterSpacing: 1,
    textTransform: "uppercase",
    marginBottom: 8,
  },
  title: {
    color: "#f8fafc",
    fontSize: 34,
    fontWeight: "900",
    letterSpacing: -1,
  },
  subtitle: {
    color: "#94a3b8",
    fontSize: 15,
    lineHeight: 22,
    marginTop: 8,
    maxWidth: 280,
  },
  logoBubble: {
    width: 58,
    height: 58,
    borderRadius: 20,
    backgroundColor: "#86efac",
    alignItems: "center",
    justifyContent: "center",
  },
  logoText: {
    color: "#020617",
    fontSize: 20,
    fontWeight: "900",
  },
  actionRow: {
    flexDirection: "row",
    gap: 10,
    marginBottom: 14,
  },
  primaryButton: {
    flex: 1,
    backgroundColor: "#22c55e",
    borderRadius: 18,
    paddingVertical: 15,
    alignItems: "center",
  },
  primaryButtonText: {
    color: "#020617",
    fontWeight: "900",
    fontSize: 15,
  },
  secondaryButton: {
    flex: 1,
    backgroundColor: "#111827",
    borderRadius: 18,
    paddingVertical: 15,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#1f2937",
  },
  secondaryButtonText: {
    color: "#e5e7eb",
    fontWeight: "800",
    fontSize: 15,
  },
  selectedCard: {
    backgroundColor: "#0f172a",
    borderRadius: 22,
    padding: 16,
    borderWidth: 1,
    borderColor: "#1e293b",
    marginBottom: 14,
  },
  cardLabel: {
    color: "#64748b",
    fontSize: 12,
    fontWeight: "900",
    textTransform: "uppercase",
    letterSpacing: 1,
  },
  selectedName: {
    color: "#f8fafc",
    fontSize: 15,
    fontWeight: "700",
    marginTop: 6,
  },
  analyzeButton: {
    backgroundColor: "#86efac",
    borderRadius: 22,
    paddingVertical: 18,
    alignItems: "center",
    marginBottom: 18,
  },
  disabledButton: {
    opacity: 0.75,
  },
  analyzeButtonText: {
    color: "#020617",
    fontSize: 17,
    fontWeight: "900",
  },
  loadingBlock: {
    alignItems: "center",
    gap: 8,
  },
  dashboardGrid: {
    gap: 14,
    marginBottom: 14,
  },
  scoreCard: {
    backgroundColor: "#0f172a",
    borderRadius: 28,
    padding: 22,
    borderWidth: 1,
    borderColor: "#1e293b",
    alignItems: "center",
  },
  scoreCircle: {
    width: 154,
    height: 154,
    borderRadius: 77,
    borderWidth: 12,
    alignItems: "center",
    justifyContent: "center",
    marginTop: 18,
    marginBottom: 14,
    backgroundColor: "#020617",
  },
  scoreBig: {
    color: "#f8fafc",
    fontSize: 42,
    fontWeight: "900",
    lineHeight: 48,
  },
  scoreSmall: {
    color: "#94a3b8",
    fontWeight: "800",
  },
  exerciseName: {
    color: "#f8fafc",
    fontSize: 23,
    fontWeight: "900",
    marginTop: 4,
  },
  confidenceText: {
    color: "#94a3b8",
    marginTop: 6,
    fontWeight: "700",
  },
  insightCard: {
    backgroundColor: "#0f172a",
    borderRadius: 28,
    padding: 20,
    borderWidth: 1,
    borderColor: "#1e293b",
  },
  bigFix: {
    color: "#f8fafc",
    fontSize: 22,
    fontWeight: "900",
    lineHeight: 29,
    marginTop: 10,
  },
  miniStats: {
    flexDirection: "row",
    gap: 10,
    marginTop: 18,
  },
  statPill: {
    flex: 1,
    backgroundColor: "#020617",
    borderRadius: 18,
    paddingVertical: 14,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#1e293b",
  },
  statNumber: {
    color: "#86efac",
    fontSize: 22,
    fontWeight: "900",
  },
  statLabel: {
    color: "#94a3b8",
    fontSize: 11,
    fontWeight: "800",
    marginTop: 3,
    textTransform: "uppercase",
  },
  card: {
    backgroundColor: "#0f172a",
    borderRadius: 28,
    padding: 18,
    borderWidth: 1,
    borderColor: "#1e293b",
    marginBottom: 14,
  },
  sectionTitle: {
    color: "#f8fafc",
    fontSize: 21,
    fontWeight: "900",
  },
  sectionSub: {
    color: "#94a3b8",
    fontSize: 13,
    fontWeight: "700",
    marginTop: 5,
    marginBottom: 14,
  },
  repRow: {
    marginTop: 14,
  },
  repTop: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 8,
  },
  repLabel: {
    color: "#e5e7eb",
    fontWeight: "900",
  },
  repScore: {
    color: "#f8fafc",
    fontWeight: "900",
  },
  repBarTrack: {
    height: 12,
    borderRadius: 999,
    backgroundColor: "#020617",
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "#1e293b",
  },
  repBarFill: {
    height: "100%",
    borderRadius: 999,
  },
  repFeedback: {
    color: "#94a3b8",
    fontSize: 13,
    marginTop: 7,
    lineHeight: 18,
  },
  coachImageWrap: {
    height: 420,
    borderRadius: 22,
    overflow: "hidden",
    backgroundColor: "#020617",
    borderWidth: 1,
    borderColor: "#1e293b",
    marginBottom: 14,
  },
  coachImage: {
    width: "100%",
    height: "100%",
    resizeMode: "contain",
  },
  emptyImage: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: 20,
  },
  emptyImageText: {
    color: "#64748b",
    fontWeight: "800",
    textAlign: "center",
  },
  zoneGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
  },
  zonePill: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    paddingHorizontal: 13,
    paddingVertical: 11,
    borderRadius: 999,
    backgroundColor: "#020617",
    borderWidth: 1,
    borderColor: "#1e293b",
  },
  zonePillActive: {
    backgroundColor: "#111827",
    borderWidth: 2,
  },
  statusDot: {
    width: 9,
    height: 9,
    borderRadius: 999,
  },
  zoneText: {
    color: "#e5e7eb",
    fontWeight: "800",
    fontSize: 13,
  },
  coachingNote: {
    marginTop: 14,
    backgroundColor: "#020617",
    borderRadius: 20,
    padding: 16,
    borderWidth: 1,
    borderColor: "#1e293b",
  },
  noteTitle: {
    color: "#f8fafc",
    fontSize: 18,
    fontWeight: "900",
  },
  noteStatus: {
    color: "#86efac",
    fontWeight: "900",
    marginTop: 6,
  },
  noteText: {
    color: "#cbd5e1",
    fontSize: 14,
    lineHeight: 21,
    marginTop: 8,
  },
  phaseScroller: {
    gap: 12,
    paddingRight: 12,
  },
  phaseCard: {
    width: "31.5%",
    minWidth: 300,
    borderRadius: 18,
    overflow: "hidden",
    backgroundColor: "#050b1a",
    borderWidth: 1,
    borderColor: "#26324a",
  },
  phaseImage: {
    width: "100%",
    height: 260,
    resizeMode: "contain",
    objectFit: "contain",
  },
  phaseCardsRow: {
    flexDirection: "row",
    gap: 20,
    flexWrap: "wrap",
    justifyContent: "space-between",
  },
  emptyPhase: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  emptyPhaseText: {
    color: "#64748b",
    fontWeight: "800",
  },
  phaseLabel: {
    color: "#f8fafc",
    fontWeight: "900",
    padding: 12,
    textAlign: "center",
  },
  warningCard: {
    backgroundColor: "#102018",
    borderRadius: 24,
    padding: 18,
    borderWidth: 1,
    borderColor: "#14532d",
    marginBottom: 14,
    gap: 8,
  },
  warningTitle: {
    color: "#bbf7d0",
    fontSize: 16,
    fontWeight: "900",
  },
  warningText: {
    color: "#dcfce7",
    lineHeight: 20,
  },
  errorCard: {
    backgroundColor: "#2a1111",
    borderRadius: 24,
    padding: 18,
    borderWidth: 1,
    borderColor: "#7f1d1d",
    marginBottom: 14,
  },
  errorTitle: {
    color: "#fecaca",
    fontSize: 17,
    fontWeight: "900",
    marginBottom: 8,
  },
  errorText: {
    color: "#fecaca",
    lineHeight: 21,
  },
  summaryLine: {
    color: "#e5e7eb",
    fontSize: 15,
    lineHeight: 22,
    marginTop: 10,
    marginBottom: 10,
  },
  feedbackLine: {
    color: "#94a3b8",
    lineHeight: 22,
    marginTop: 4,
  },
  overlayButton: {
    backgroundColor: "#22c55e",
    borderRadius: 18,
    paddingVertical: 16,
    alignItems: "center",
    marginBottom: 14,
  },
  overlayButtonDisabled: {
    backgroundColor: "#374151",
    borderRadius: 18,
    paddingVertical: 16,
    alignItems: "center",
    marginBottom: 14,
    opacity: 0.7,
  },
  overlayCard: {
    backgroundColor: "#111827",
    borderRadius: 28,
    padding: 18,
    marginBottom: 18,
    borderWidth: 1,
    borderColor: "#243044",
    overflow: "visible",
  },
  videoPlayer: {
    width: "100%",
    height: 620,
    borderRadius: 22,
    backgroundColor: "#020617",
  },
  videoShell: {
    width: "100%",
    borderRadius: 22,
    overflow: "visible",
    backgroundColor: "#020617",
  },
  coachText: {
    color: "#E5E7EB",
    fontSize: 16,
    lineHeight: 22,
    marginTop: 8,
  },
});
